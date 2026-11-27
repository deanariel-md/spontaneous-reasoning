"""Size ladder: four instruct checkpoints, thinking off, run with the main panel's complete design on GPU machines.

Models: Llama-3.1-70B-Instruct (FP8, 1 card; the release of the panel's
Llama-3.1-8B-Instruct), Qwen3.5-122B-A10B (Qwen's FP8, tensor parallel 2 on 2 cards; the release of the panel's
Qwen3.5-9B and 35B-A3B), Ministral 3 8B and 14B Instruct 2512 (bfloat16, 1 card each; the release of Mistral Large 3).
Qwen3-235B, gpt-oss-120b, DeepSeek V4-Flash-0731 and Llama-3.3-70B are not part of the ladder (DROPPED); their outputs
are not analysed.

Design. Each ladder model answers exactly the main panel's request rows, model_key replaced (checked against a second
panel model's files): per benchmark the requested (direct_answer), chain-of-thought (zero_shot_cot) and forced
(forced_direct) arms, and the five-shot arms that the panel has on that benchmark:
  GSM8K and BIG-Bench Hard  five-shot forced and five-shot requested (8 configurations);
  MATH-500                  five-shot forced;
  HotpotQA                  five-shot forced (blocks A and B).
The panel's eight-shot chain of thought (GSM8K) is not a five-shot arm and is not part of the ladder.

Caps. Requested, chain-of-thought and five-shot requested arms run at RAISED_CAP (7,000) new tokens; each
output is also kept cut at the token level to the panel's cap of its arm (Decoding.cut_caps: 1,024 requested; 3,072
chain of thought, 1,900 on HotpotQA), marked cut when the uncut output ran past it. The forced arms keep the panel's
caps (16; 64 on HotpotQA). Every other decoding parameter equals the arm's panel collection (MAIN_DECODING).

Registry: config/models_ladder.yaml. Requests: data/requests/ladder/<task>/<key>.parquet and the pilot
data/requests/ladder/pilot/<key>.parquet (scripts/build_ladder_requests.py). Every ladder job reads its
requests from a shared code copy and writes to $S/out/ladder/<run key>/<task>.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "config/models_ladder.yaml"
TASKS = ("gsm8k", "math", "bbh", "hotpot")
TASK_ARMS: dict[str, tuple[str, ...]] = {
    "gsm8k": ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced", "fewshot_requested"),
    "math": ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced"),
    "bbh": ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced", "fewshot_requested"),
    "hotpot": ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced"),
}
ARMS = ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced", "fewshot_requested")
FORCED_ARMS = ("forced_direct", "fewshot_forced")
RAISED_ARMS = ("direct_answer", "zero_shot_cot", "fewshot_requested")
RAISED_CAP = 7000
SOURCE_KEY = "llama-3.1-8b-instruct"      # any panel model: the files differ only in model_key
CHECK_KEY = "gemma-2b-it"                 # a second model whose files must hold the same requests
# task -> list of (request file pattern, {arm in the file: ladder arm}, request_id prefix kept or None)
SOURCES: dict[str, list[tuple[str, dict[str, str], str | None]]] = {
    "gsm8k": [("data/requests/gsm8k/{k}.parquet", {"direct_answer": "direct_answer", "zero_shot_cot": "zero_shot_cot"}, None),
              ("data/requests/forced/gsm8k/{k}.parquet", {"forced_direct": "forced_direct"}, None),
              ("data/requests/fewshot/gsm8k/{k}.parquet", {"forced_direct": "fewshot_forced"}, None),
              ("data/requests/fewshot_requested/collection/{k}.parquet", {"direct_answer": "fewshot_requested"}, "gsm8k/")],
    "math": [("data/requests/math/{k}.parquet", {"direct_answer": "direct_answer", "zero_shot_cot": "zero_shot_cot"}, None),
             ("data/requests/forced/math/{k}.parquet", {"forced_direct": "forced_direct"}, None),
             ("data/requests/fewshot/math/{k}.parquet", {"forced_direct": "fewshot_forced"}, None)],
    "bbh": [("data/requests/bbh/{k}.parquet", {"direct_answer": "direct_answer", "zero_shot_cot": "zero_shot_cot"}, None),
            ("data/requests/forced/bbh/{k}.parquet", {"forced_direct": "forced_direct"}, None),
            ("data/requests/fewshot/bbh/{k}.parquet", {"forced_direct": "fewshot_forced"}, None),
            ("data/requests/fewshot_requested/collection/{k}.parquet", {"direct_answer": "fewshot_requested"}, "bbh/")],
    "hotpot": [("data/requests/hotpot/block_a/{k}.parquet", {a: a for a in TASK_ARMS["hotpot"]}, None),
               ("data/requests/hotpot/block_b/{k}.parquet", {a: a for a in TASK_ARMS["hotpot"]}, None)],
}
DECODING = {t: f"config/decoding_ladder2_{t}.yaml" for t in TASKS}
PILOT_DECODING = "config/decoding_ladder2_pilot.yaml"
EQUIV_DECODING = "config/decoding_ladder2_equiv.yaml"
# The decoding (file, arm in that file) of each ladder arm's panel collection. params_for_arm of the ladder arm equals
# that one's except max_new_tokens, which is RAISED_CAP for RAISED_ARMS (tests/test_ladder.py).
MAIN_DECODING: dict[str, dict[str, tuple[str, str]]] = {
    "gsm8k": {"direct_answer": ("config/decoding.yaml", "direct_answer"),
              "zero_shot_cot": ("config/decoding.yaml", "zero_shot_cot"),
              "forced_direct": ("config/decoding_forced.yaml", "forced_direct"),
              "fewshot_forced": ("config/decoding_forced.yaml", "forced_direct"),
              "fewshot_requested": ("config/decoding.yaml", "direct_answer")},
    "math": {"direct_answer": ("config/decoding.yaml", "direct_answer"),
             "zero_shot_cot": ("config/decoding.yaml", "zero_shot_cot"),
             "forced_direct": ("config/decoding_forced_math.yaml", "forced_direct"),
             "fewshot_forced": ("config/decoding_forced_math.yaml", "forced_direct")},
    "bbh": {"direct_answer": ("config/decoding.yaml", "direct_answer"),
            "zero_shot_cot": ("config/decoding.yaml", "zero_shot_cot"),
            "forced_direct": ("config/decoding_forced.yaml", "forced_direct"),
            "fewshot_forced": ("config/decoding_forced.yaml", "forced_direct"),
            "fewshot_requested": ("config/decoding.yaml", "direct_answer")},
    "hotpot": {a: ("config/decoding_hotpot.yaml", a) for a in TASK_ARMS["hotpot"]},
}
PILOT_ITEMS = 16          # per task and arm: the first 16 items of one configuration
EQUIV_ROWS = 50           # rows of the equivalence check

# Cards and pools: Qwen3.5-122B on 2 cards of the reserved pool; the one-card models on cards of the public pool.
# Job names d2-ld-<short>-<task>[-p<k>] and d2e-ld-<short>-<task>[-p<k>].
MODELS: dict[str, dict[str, Any]] = {
    "llama-3.1-70b-instruct-fp8": {"pool": "public", "cards": 1, "mem": "96G", "short": "llama70",
                                   "family": "Llama", "ladder": "Llama-3.1 8B -> 70B", "precision": "FP8 (RedHatAI FP8-dynamic of Meta's release; KV cache 16-bit)"},
    "qwen3.5-122b-a10b-fp8": {"pool": "reserved", "cards": 2, "mem": "180G", "short": "qwen122",
                              "family": "Qwen", "ladder": "Qwen3.5 9B -> 35B-A3B -> 122B-A10B", "precision": "FP8 (Qwen's official block FP8; KV cache 16-bit)"},
    "ministral-3-8b-instruct-2512": {"pool": "public", "cards": 1, "mem": "64G", "short": "min8",
                                     "family": "Mistral", "ladder": "Ministral 3 8B -> 14B -> Large 3", "precision": "bfloat16"},
    "ministral-3-14b-instruct-2512": {"pool": "public", "cards": 1, "mem": "96G", "short": "min14",
                                      "family": "Mistral", "ladder": "Ministral 3 8B -> 14B -> Large 3", "precision": "bfloat16"},
}
FALLBACK_SUFFIX = "-dk"   # the registry key with vLLM's default kernels
EQUIV_ORDER = ("llama-3.1-70b-instruct-fp8", "ministral-3-14b-instruct-2512", "ministral-3-8b-instruct-2512")
DROPPED: dict[str, str] = {   # key -> why it is not part of the ladder; outputs not analysed
    "qwen3-235b-a22b-instruct-2507-fp8": "dropped from the ladder (Qwen3-235B)",
    "gpt-oss-120b": "dropped from the ladder (gpt-oss-120b and gpt-oss-20b)",
    "deepseek-v4-flash-0731": "dropped from the ladder (DeepSeek V4-Flash-0731 on the GPU machines)",
    "llama-3.3-70b-instruct-fp8": "dropped from the ladder (Llama-3.1-70B is used instead)",
}

# --------------------------------------------------------------------------------------------
# Time estimate, written before any output of these models exists; the pilot replaces it.
# Output lengths: a same-family panel model (PROXY; results/ladder/proxy_lengths.json, read from the panel's outputs),
# with every output the panel cut at its cap counted at the raised cap (an upper bound). Prompt tokens: the proxy's mean
# prompt length per arm plus the model's own system-message overhead (Ministral's default system message).
# Rates (output tokens and prompt tokens per second per job), from panel runs on the same cards: Mistral-Small-3.1-24B
# (batch-invariant, RTX PRO 6000) about 1,500 output and 6,000 prompt tokens per second; Qwen3.5-35B-A3B (default kernels)
# about 2,500 and 27,000; scaled by weight size for the dense models and by active parameters over two cards for Qwen.
# --------------------------------------------------------------------------------------------
PROXY = {"llama-3.1-70b-instruct-fp8": "llama-3.1-8b-instruct", "qwen3.5-122b-a10b-fp8": "qwen3.5-35b-a3b",
         "ministral-3-8b-instruct-2512": "mistral-nemo-instruct-2407",
         "ministral-3-14b-instruct-2512": "mistral-small-3.1-24b-instruct-2503"}
RATES = {   # (output tok/s, prompt tok/s, model load hours)
    "llama-3.1-70b-instruct-fp8": (600, 2500, 0.15),
    "qwen3.5-122b-a10b-fp8": (1800, 15000, 0.3),
    "ministral-3-8b-instruct-2512": (2500, 12000, 0.1),
    "ministral-3-14b-instruct-2512": (1800, 8000, 0.1),
}
SYSTEM_OVERHEAD = {"ministral-3-8b-instruct-2512": 520, "ministral-3-14b-instruct-2512": 520}
# (MATH-500's forced arm has no local proxy table; GSM8K's forced lengths stand in, a few tokens either way.)
PROXY_FILES = {"gsm8k": {"direct_answer": ("gsm8k", "direct_answer"), "zero_shot_cot": ("gsm8k", "zero_shot_cot"),
                         "forced_direct": ("gsm8k_forced", "forced_direct"),
                         "fewshot_forced": ("gsm8k_fewshot", "forced_direct"), "fewshot_requested": ("fsr", "direct_answer")},
               "math": {"direct_answer": ("math", "direct_answer"), "zero_shot_cot": ("math", "zero_shot_cot"),
                        "forced_direct": ("gsm8k_forced", "forced_direct"), "fewshot_forced": ("math_fewshot", "forced_direct")},
               "bbh": {"direct_answer": ("bbh", "direct_answer"), "zero_shot_cot": ("bbh", "zero_shot_cot"),
                       "forced_direct": ("bbh_forced", "forced_direct"), "fewshot_forced": ("bbh_fewshot", "forced_direct"),
                       "fewshot_requested": ("fsr", "direct_answer")},
               "hotpot": {a: ("hotpot", a) for a in TASK_ARMS["hotpot"]}}
REQUESTS = {   # per task and arm, the panel's request counts
    "gsm8k": {"direct_answer": 38251, "zero_shot_cot": 38251, "forced_direct": 38251, "fewshot_forced": 38251,
              "fewshot_requested": 10552},
    "math": {"direct_answer": 12338, "zero_shot_cot": 12338, "forced_direct": 12338, "fewshot_forced": 12338},
    "bbh": {"direct_answer": 47250, "zero_shot_cot": 47250, "forced_direct": 47250, "fewshot_forced": 47250,
            "fewshot_requested": 10800},
    "hotpot": {a: 36000 for a in TASK_ARMS["hotpot"]},
}
TARGET_PART_HOURS = 6.0   # one-card parts of at most about 6 h, so each ends well inside 12 h and parts run side by side
PANEL_CAPS = {"gsm8k": {"direct_answer": 1024, "zero_shot_cot": 3072, "fewshot_requested": 1024},
              "math": {"direct_answer": 1024, "zero_shot_cot": 3072},
              "bbh": {"direct_answer": 1024, "zero_shot_cot": 3072, "fewshot_requested": 1024},
              "hotpot": {"direct_answer": 1024, "zero_shot_cot": 1900}}


PROXY_PATH = ROOT / "results" / "ladder" / "proxy_lengths.json"


def proxy_lengths(path: Path | None = None) -> dict:
    return json.loads((path or PROXY_PATH).read_text(encoding="utf-8"))


def arm_tokens(key: str, task: str, arm: str, proxy: Mapping | None = None) -> tuple[float, float]:
    """(mean output tokens at the raised cap, mean prompt tokens) per request, from the proxy model."""
    proxy = proxy or proxy_lengths()
    src, src_arm = PROXY_FILES[task][arm]
    r = proxy[PROXY[key]][src][src_arm]
    out = float(r["mean_out"])
    cap = PANEL_CAPS[task].get(arm)
    if cap:
        out += float(r["cut"]) * (RAISED_CAP - cap)
    return out, float(r["mean_prompt"]) + SYSTEM_OVERHEAD.get(key, 0)


def wall_hours(key: str, task: str, proxy: Mapping | None = None) -> float:
    proxy = proxy or proxy_lengths()
    r_out, r_in, _ = RATES[key]
    out = sum(REQUESTS[task][a] * arm_tokens(key, task, a, proxy)[0] for a in TASK_ARMS[task])
    pro = sum(REQUESTS[task][a] * arm_tokens(key, task, a, proxy)[1] for a in TASK_ARMS[task])
    return out / r_out / 3600 + pro / r_in / 3600


def parts(key: str, task: str, proxy: Mapping | None = None) -> int:
    if MODELS[key]["pool"] != "public":
        return 1          # the reserved-pool model runs its tasks one after another on its own cards
    return max(1, math.ceil(wall_hours(key, task, proxy) / TARGET_PART_HOURS))


def estimate(key: str, proxy: Mapping | None = None) -> dict:
    proxy = proxy or proxy_lengths()
    rows = {}
    for t in TASKS:
        n = parts(key, t, proxy)
        w = wall_hours(key, t, proxy) + n * RATES[key][2]
        rows[t] = {"wall_hours": round(w, 1), "parts": n, "gpu_hours": round(w * MODELS[key]["cards"], 1)}
    wall = sum(r["wall_hours"] for r in rows.values())
    return {"tasks": rows, "wall_hours": round(wall, 1), "gpu_hours": round(wall * MODELS[key]["cards"], 1)}


def jobs(key: str, proxy: Mapping | None = None) -> list[dict]:
    """The model's full jobs: one per task, or one per part of a task (TASK, PART, job name stem)."""
    out = []
    for t in TASKS:
        n = parts(key, t, proxy)
        for k in range(1, n + 1):
            stem = f"{MODELS[key]['short']}-{t}" + (f"-p{k}" if n > 1 else "")
            out.append({"task": t, "part": f"{k}/{n}" if n > 1 else "", "stem": stem})
    return out


# --------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------

def _sig(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for rid, m in sorted(zip(df["request_id"], df["messages_json"])):
        h.update(rid.encode("utf-8") + b"\0" + m.encode("utf-8") + b"\n")
    return h.hexdigest()


def _select(df: pd.DataFrame, arms: Mapping[str, str], prefix: str | None) -> pd.DataFrame:
    df = df[df["arm"].isin(list(arms))]
    if prefix is not None:
        df = df[df["request_id"].astype(str).str.startswith(prefix)]
    return df


def task_requests(task: str, key: str, root: Path = ROOT, source_key: str = SOURCE_KEY,
                  check_key: str | None = CHECK_KEY) -> tuple[pd.DataFrame, dict]:
    """The task's arms for ladder model `key`: the panel's rows with model_key replaced and the five-shot arms renamed
    (fewshot_forced, fewshot_requested), nothing else changed. With check_key, each source's rows must equal that
    model's (request ids and messages)."""
    frames, record = [], {"sources": []}
    for pattern, arms, prefix in SOURCES[task]:
        src = root / pattern.format(k=source_key)
        df = _select(pd.read_parquet(src), arms, prefix).copy()
        rec = {"path": pattern.format(k=source_key), "rows": int(len(df)), "arms": dict(arms), "prefix": prefix,
               "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
        if check_key:
            other = _select(pd.read_parquet(root / pattern.format(k=check_key)), arms, prefix)
            if _sig(other) != _sig(df):
                raise ValueError(f"{pattern}: {source_key}'s requests differ from {check_key}'s")
            rec["equal_to"] = pattern.format(k=check_key)
        if df.empty:
            raise ValueError(f"{pattern}: no rows for arms {sorted(arms)}{' with prefix ' + prefix if prefix else ''}")
        df["arm"] = df["arm"].map(arms)
        record["sources"].append(rec)
        frames.append(df)
    for f in frames:   # GSM8K's item ids are integers in its panel tables; the other tasks' are strings
        f["item_id"] = f["item_id"].astype("int64") if task == "gsm8k" else f["item_id"].astype(str)
    out = pd.concat(frames, ignore_index=True)
    if out["request_id"].duplicated().any():
        raise ValueError(f"{task}: request ids repeat across source files")
    missing = sorted(set(TASK_ARMS[task]) - set(out["arm"]))
    if missing:
        raise ValueError(f"{task}: arms {missing} absent")
    out["model_key"] = key
    record.update(rows=int(len(out)), by_arm={a: int(v) for a, v in out["arm"].value_counts().sort_index().items()},
                  configs=int(out["config_id"].nunique()), items=int(out["item_id"].astype(str).nunique()),
                  configs_by_arm={a: int(g["config_id"].nunique()) for a, g in out.groupby("arm")})
    return out, record


def pilot_rows(task: str, df: pd.DataFrame, n_items: int = PILOT_ITEMS) -> pd.DataFrame:
    """The pilot's rows of one task: per arm, the configuration of the file's first direct_answer row (or, for an arm
    without rows there, the arm's own first configuration) and its first n_items items in file order. Arms are
    relabelled <task>.<arm> and request ids prefixed <task>|, so that one pilot run holds all four tasks under
    config/decoding_ladder2_pilot.yaml."""
    cfg0 = df[df["arm"] == "direct_answer"].iloc[0]["config_id"]
    parts_ = []
    for arm in TASK_ARMS[task]:
        sub = df[df["arm"] == arm]
        cfg = cfg0 if (sub["config_id"] == cfg0).any() else sub.iloc[0]["config_id"]
        sub = sub[sub["config_id"] == cfg]
        items = list(dict.fromkeys(sub["item_id"].tolist()))[:n_items]
        parts_.append(sub[sub["item_id"].isin(items)])
    rows = pd.concat(parts_).copy()
    rows["request_id"] = task + "|" + rows["request_id"].astype(str)
    rows["arm"] = task + "." + rows["arm"]
    rows["item_id"] = rows["item_id"].astype(str)
    return rows


def _arm_params(raw: Mapping, arm: str) -> dict:
    ov = dict((raw.get("arm_overrides") or {}).get(arm) or {})
    return {"stop": list(ov.get("stop", raw.get("stop") or [])),
            "include_stop_str_in_output": bool(ov.get("include_stop_str_in_output", raw.get("include_stop_str_in_output", False)))}


def task_decoding_dict(task: str, root: Path = ROOT) -> dict:
    """The ladder decoding of one task: each arm with its panel collection's parameters (MAIN_DECODING), the raised
    cap on RAISED_ARMS with the panel's cap as its cut cap."""
    import yaml
    out: dict[str, Any] = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
                           "max_new_tokens": {}, "prefill": {}, "stop": [], "include_stop_str_in_output": False,
                           "arm_overrides": {}, "cut_caps": {}, "skip_special_tokens": True}
    engine = None
    for arm in TASK_ARMS[task]:
        f, src_arm = MAIN_DECODING[task][arm]
        raw = yaml.safe_load(open(root / f, encoding="utf-8"))
        for k in ("temperature", "top_p", "top_k", "seed", "skip_special_tokens"):
            if raw.get(k, out[k]) != out[k]:
                raise ValueError(f"{f}: {k} differs from the other arms'")
        engine = engine or raw["engine"]
        cap = int(raw["max_new_tokens"][src_arm])
        if arm in RAISED_ARMS:
            out["max_new_tokens"][arm] = RAISED_CAP
            out["cut_caps"][arm] = cap
        else:
            out["max_new_tokens"][arm] = cap
        if (raw.get("prefill") or {}).get(src_arm):
            out["prefill"][arm] = raw["prefill"][src_arm]
        p = _arm_params(raw, src_arm)
        if p["stop"] or p["include_stop_str_in_output"]:
            out["arm_overrides"][arm] = p
    out["engine"] = engine
    return out


def pilot_decoding_dict(root: Path = ROOT, equiv: bool = False) -> dict:
    """One decoding with an arm <task>.<arm> per task and arm whose parameters equal the task's ladder decoding. With
    equiv, only the raised arms, each at the panel's cap and without a cut (the equivalence run)."""
    out: dict[str, Any] = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
                           "max_new_tokens": {}, "prefill": {}, "stop": [], "include_stop_str_in_output": False,
                           "arm_overrides": {}, "cut_caps": {}, "skip_special_tokens": True}
    engine = None
    for t in TASKS:
        d = task_decoding_dict(t, root)
        engine = engine or d["engine"]
        for a in TASK_ARMS[t]:
            arm = f"{t}.{a}"
            if equiv:
                if a in d["cut_caps"]:
                    out["max_new_tokens"][arm] = d["cut_caps"][a]
                continue
            out["max_new_tokens"][arm] = d["max_new_tokens"][a]
            if a in d["prefill"]:
                out["prefill"][arm] = d["prefill"][a]
            if a in d["arm_overrides"]:
                out["arm_overrides"][arm] = d["arm_overrides"][a]
            if a in d["cut_caps"]:
                out["cut_caps"][arm] = d["cut_caps"][a]
    for k in ("prefill", "arm_overrides", "cut_caps"):
        if not out[k]:
            out.pop(k)
    out["engine"] = engine
    return out


def equiv_rows(pilot_requests: pd.DataFrame, pilot_out: pd.DataFrame, n: int = EQUIV_ROWS) -> pd.DataFrame:
    """The equivalence check's requests, from the pilot of the raised arms: every row whose
    output the pilot cut at the panel's cap, in pilot file order, then the longest other outputs (most output tokens,
    ties by request id), up to n rows. The rule reads output lengths only."""
    raised = pilot_out[pilot_out["arm"].str.split(".").str[1].isin(RAISED_ARMS)].copy()
    order = {r: i for i, r in enumerate(pilot_requests["request_id"])}
    raised["_o"] = raised["request_id"].map(order)
    cut = raised[raised["cut_at_panel_cap"].astype(bool)].sort_values("_o")
    rest = raised[~raised["cut_at_panel_cap"].astype(bool)].sort_values(["n_output_tokens", "request_id"],
                                                                         ascending=[False, True])
    ids = (list(cut["request_id"]) + list(rest["request_id"]))[:n]
    return pilot_requests.set_index("request_id").loc[ids].reset_index()


# --------------------------------------------------------------------------------------------
# Pilot check (no correctness is read)
# --------------------------------------------------------------------------------------------

_BOX = "\\boxed{"


def boxed_answer(text: str) -> str | None:
    """Content of the last complete \\boxed{...} in text (balanced braces), or None."""
    if not isinstance(text, str):
        return None
    i = text.rfind(_BOX)
    while i >= 0:
        j, depth = i + len(_BOX), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            return text[i + len(_BOX): j - 1]
        i = text.rfind(_BOX, 0, i)
    return None


def forced_closed(text: str) -> bool:
    """A forced continuation (after the prefill \\boxed{) closes the box with a non-empty answer before the brace."""
    a = boxed_answer(_BOX + (text or ""))
    return a is not None and a.strip() != ""


def arm_class(arm: str) -> str:
    return "forced" if arm in FORCED_ARMS else ("cot" if arm == "zero_shot_cot" else "requested")


def pilot_summary(df: pd.DataFrame) -> dict:
    """Per task and arm: rows, box closed (forced arms) or a boxed answer present (the others, uncut outputs), outputs
    past the panel's cap, at the raised cap, and identical outputs to different prompts."""
    res: dict[str, Any] = {}
    for arm_label, g in df.groupby("arm"):
        task, arm = arm_label.split(".", 1)
        texts = g["output_text"].fillna("")
        ok = texts.map(forced_closed) if arm in FORCED_ARMS else texts.map(lambda s: boxed_answer(s) is not None)
        dup = g[g.duplicated("output_text", keep=False) & ~g.duplicated("prompt_sha256", keep=False)]
        r = {"task": task, "arm": arm, "rows": int(len(g)), "pass": int(ok.sum()),
             "rate": round(float(ok.mean()), 4) if len(g) else None,
             "cut": int((g["finish_reason"] == "length").sum()),
             "mean_output_tokens": round(float(g["n_output_tokens"].mean()), 1),
             "max_output_tokens": int(g["n_output_tokens"].max()),
             "identical_outputs_to_different_prompts": int(len(dup))}
        if "cut_at_panel_cap" in g:
            r["past_panel_cap"] = int(g["cut_at_panel_cap"].astype(bool).sum())
            r["mean_output_tokens_cut"] = round(float(g["n_output_tokens_cut"].mean()), 1)
        res[arm_label] = r
    return res


def pilot_verdict(summary: Mapping[str, Mapping], expected_rows: int, rows: int) -> tuple[bool, list[str]]:
    """Pass when every pilot row has an output, the forced arms (zero- and five-shot) together close the box in at
    least 90% of rows, and the requested arms (zero- and five-shot) and the chain-of-thought arm each give a boxed
    answer in at least 90% of rows, pooled over tasks."""
    why = []
    if rows != expected_rows:
        why.append(f"{rows} of {expected_rows} pilot rows have outputs")

    def pooled(cls):
        s = [v for v in summary.values() if arm_class(v["arm"]) == cls]
        n = sum(v["rows"] for v in s)
        return (sum(v["pass"] for v in s) / n) if n else 0.0
    f = pooled("forced")
    if f < 0.9:
        why.append(f"forced arms: box closed in {f:.3f} of rows (< 0.9)")
    for cls in ("requested", "cot"):
        r = pooled(cls)
        if r < 0.9:
            why.append(f"{cls} arms: boxed answer in {r:.3f} of rows (< 0.9)")
    return not why, why


def kv_cache_lines(log_text: str) -> list[str]:
    """vLLM's lines on KV cache size and concurrency from a job's stdout."""
    pat = re.compile(r"(GPU KV cache size|Maximum concurrency|Available KV cache memory|init engine)", re.IGNORECASE)
    return [ln.strip()[-200:] for ln in log_text.splitlines() if pat.search(ln)][:12]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
