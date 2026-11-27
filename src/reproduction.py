r"""Reproduction of Cheng et al. (2025), GSM8K zero-shot versus eight-exemplar chain of thought.

Data flow
    config/reproduction.yaml + manifests/cheng_exemplars.json (+ the frozen bank for transport)
    -> request files (scripts/build_reproduction_requests.py), one per model and part
    -> src/generation.py runner (scripts/generate_d2.py with --models-yaml config/models_reproduction.yaml and the
       model's --decoding-yaml)
    -> scored table (scripts/score_reproduction.py) -> results/reproduction/ (scripts/run_reproduction.py)

Parts. "numerical" renders the authors' published frames (config_id "cheng"); "transport" renders the eight tranche
configurations of the frozen Dataset 2 bank in the same two arms. Request ids are f"{config_id}|{arm}|{item_id}" on
the GSM8K test items, the runner's columns (generation.REQUEST_COLUMNS).

Arms. zero_shot_cot: the frame with the test question. eight_shot_cot: eight user/assistant turn pairs (the frame
filled with each exemplar question, then the authors' exemplar answer unchanged) and the frame with the test question.
The item variable is the literal token {question}, replaced by str.replace exactly once per message list, as in
src/d2_io.py.

Scorer. cheng_extract and cheng_is_equal follow opencompass/datasets/gsm8k.py at the authors' commit 40de49b9006c
(gsm8k_postprocess_fixed, last_boxed_only_string, remove_boxed, extract_boxed_answer, Gsm8kEvaluator.is_equal) line
for line; cheng_reference is their gsm8k_dataset_postprocess.

Units: accuracies and benefits in percentage points. Benefit = eight-shot minus zero-shot accuracy on the same items.
Intervals: Wilson score for one accuracy, Newcombe's method 10 for the paired difference. Nothing is clipped: a
benefit of exactly zero is a tie, and an undefined quantity is NaN with the reason recorded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from . import generation as gen

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "reproduction.yaml"
ARMS = ("zero_shot_cot", "eight_shot_cot")
NUMERICAL_ID = "cheng"
PARTS = ("numerical", "transport")
SCORER_VERSION = "cheng-fixed/1.0.0"
QUESTION = "{question}"
REQUESTS_DIR = ROOT / "data" / "requests" / "reproduction"


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("status") != "fixed":
        raise ValueError("config/reproduction.yaml is not fixed")
    if tuple(cfg.get("arms") or ()) != ARMS:
        raise ValueError(f"arms must be {ARMS}")
    return cfg


def load_json(rel: str | Path) -> dict:
    p = Path(rel)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Prompts and requests
# --------------------------------------------------------------------------------------------

def fill(messages: Sequence[Mapping[str, str]], question: str) -> list[dict]:
    """Copy of the messages with {question} replaced once; refuses a list where it occurs other than once."""
    n = sum(m["content"].count(QUESTION) for m in messages)
    if n != 1:
        raise ValueError(f"{QUESTION} occurs {n} times in the messages; exactly once is required")
    return [{"role": m["role"], "content": m["content"].replace(QUESTION, question)} for m in messages]


def eight_shot(frame: Sequence[Mapping[str, str]], exemplars: Sequence[Mapping[str, str]]) -> list[dict]:
    """Eight user/assistant pairs from a one-turn user frame, then the frame itself (still holding {question})."""
    if len(frame) != 1 or frame[0]["role"] != "user":
        raise ValueError("an eight-shot arm needs a frame of exactly one user message")
    if len(exemplars) != 8:
        raise ValueError(f"expected 8 exemplars, found {len(exemplars)}")
    out: list[dict] = []
    for ex in exemplars:
        out += fill(frame, ex["question"])
        out.append({"role": "assistant", "content": ex["answer"]})
    return out + [dict(frame[0])]


def published_arms(prompts: Mapping) -> dict[str, list[dict]]:
    """The authors' frames (Fig. 16; gsm8k_gen_zo_v2_fixed.py and gsm8k_gen_original_v2-fixed.py)."""
    return {"zero_shot_cot": [{"role": "user", "content": prompts["zero_shot_user"]}],
            "eight_shot_cot": eight_shot([{"role": "user", "content": prompts["eight_shot_user"]}], prompts["exemplars"])}


def transport_arms(bank: Mapping, config_id: str, prompts: Mapping) -> dict[str, list[dict]]:
    """A bank configuration's chain-of-thought messages as the zero-shot frame, and the eight-shot arm built on it."""
    frame = copy.deepcopy(bank["configurations"][config_id]["arms"]["zero_shot_cot"]["messages"])
    if "\\boxed{" not in frame[-1]["content"]:
        raise ValueError(f"{config_id}: the frame does not ask for a boxed answer")
    return {"zero_shot_cot": frame, "eight_shot_cot": eight_shot(frame, prompts["exemplars"])}


def part_arms(part: str, cfg: Mapping, prompts: Mapping, bank: Mapping | None = None) -> dict[str, dict[str, list[dict]]]:
    """{config_id: {arm: messages}} for one part."""
    if part == "numerical":
        return {NUMERICAL_ID: published_arms(prompts)}
    if part == "transport":
        tranche = load_json(cfg["parts"]["transport"]["configurations_from"])
        bank = bank if bank is not None else load_json(cfg["parts"]["transport"]["bank"])
        return {c: transport_arms(bank, c, prompts) for c in tranche["configurations"]}
    raise ValueError(f"part must be one of {PARTS}")


def build_requests(model_key: str, arms_by_config: Mapping[str, Mapping[str, list[dict]]],
                   items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, arms in arms_by_config.items():
        if "|" in cid:
            raise ValueError(f"config_id {cid!r} contains '|'")
        for arm in ARMS:
            for iid, q in zip(items["item_id"].tolist(), items["question"].tolist()):
                rows.append({"request_id": f"{cid}|{arm}|{iid}", "model_key": model_key, "arm": arm, "config_id": cid,
                             "item_id": int(iid), "messages_json": json.dumps(fill(arms[arm], q), ensure_ascii=False)})
    df = pd.DataFrame(rows, columns=list(gen.REQUEST_COLUMNS))
    df["item_id"] = df["item_id"].astype("int64")
    if df["request_id"].duplicated().any():
        raise ValueError("duplicate request ids")
    return df


# --------------------------------------------------------------------------------------------
# Scorer: the authors' corrected extractor and match rule (opencompass/datasets/gsm8k.py, commit 40de49b9006c)
# --------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"\-?\d+\.\d+|\-?\d+")


def _last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def _remove_boxed(s: str) -> str | None:
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1:] == "}":
        return s[len(left):-1]
    return None


def _extract_boxed_answer(pred_str: str, strip_double_curly_brace: bool = False) -> str | None:
    boxed_str = _last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = _remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match(r"^\{(.*)\}$", answer)
        if match:
            answer = match.group(1)
    return answer


def cheng_extract(text: str) -> str:
    """gsm8k_postprocess_fixed: the last number in the last box, else the last number before 'Question:', else NULL."""
    text = text.replace(",", "")
    cand_ans = _extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        numbers = _NUMBER.findall(cand_ans)
        return numbers[-1] if numbers else "NULL"
    text = text.split("Question:")[0]
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else "NULL"


def cheng_reference(gold_text: str) -> str:
    """gsm8k_dataset_postprocess: the text after '#### ' with commas removed."""
    return gold_text.split("#### ")[1].replace(",", "")


def cheng_is_equal(pred: str, refer: str) -> bool:
    """Gsm8kEvaluator.is_equal."""
    try:
        if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
            return True
    except Exception:  # noqa: BLE001 - the authors' rule: anything unparsable is wrong
        pass
    return False


def score_outputs(outputs: pd.DataFrame, items: pd.DataFrame, part: str) -> pd.DataFrame:
    """One row per output: the extracted prediction, correctness, and the NULL and truncation flags."""
    ref = dict(zip(items["item_id"].astype(int), (cheng_reference(t) for t in items["gold_text"])))
    pred = [cheng_extract(t) for t in outputs["output_text"]]
    out = pd.DataFrame({
        "model_key": outputs["model_key"].astype(str).to_numpy(), "part": part,
        "config_id": outputs["config_id"].astype(str).to_numpy(), "arm": outputs["arm"].astype(str).to_numpy(),
        "item_id": outputs["item_id"].astype("int64").to_numpy(), "pred": pred,
        "reference": [ref[int(i)] for i in outputs["item_id"]],
        "n_output_tokens": outputs["n_output_tokens"].astype("int64").to_numpy(),
        "truncated": (outputs["finish_reason"].astype(str) == "length").to_numpy(),
        "cached": outputs["cached"].astype(bool).to_numpy()})
    out["correct"] = [int(cheng_is_equal(p, r)) for p, r in zip(out["pred"], out["reference"])]
    out["null_pred"] = out["pred"] == "NULL"
    out["scorer"] = SCORER_VERSION
    return out


# --------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------

def _z(level: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + level / 2)


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes in n, as proportions; (nan, nan) when n is 0."""
    if n == 0:
        return math.nan, math.nan
    z = _z(level)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def newcombe_paired(a: int, b: int, c: int, d: int, level: float = 0.95) -> dict:
    """Newcombe (1998) method 10 for p1 - p2 from paired outcomes: a both right, b only arm 1 right, c only arm 2
    right, d neither. p1 = (a+b)/n, p2 = (a+c)/n. phi is 0 when its denominator is 0 (recorded)."""
    n = a + b + c + d
    if n == 0:
        return {"diff": math.nan, "lo": math.nan, "hi": math.nan, "phi": math.nan, "phi_note": "n is 0"}
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, level)
    l2, u2 = wilson(a + c, n, level)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi, note = ((a * d - b * c) / math.sqrt(den), None) if den > 0 else (0.0, "a margin is empty; phi set to 0")
    theta = p1 - p2
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return {"diff": theta, "lo": theta - dl, "hi": theta + du, "phi": phi, "phi_note": note}


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------

def accuracy_table(scored: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    rows = []
    for (m, part, cid, arm), g in scored.groupby(["model_key", "part", "config_id", "arm"], sort=True):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson(k, n, level)
        rows.append({"model_key": m, "part": part, "config_id": cid, "arm": arm, "n": n, "correct": k,
                     "accuracy": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi, "null_preds": int(g["null_pred"].sum()),
                     "truncated": int(g["truncated"].sum())})
    return pd.DataFrame(rows)


def benefit_table(scored: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Per model and configuration: paired counts over common items, accuracies, benefit and its interval."""
    rows = []
    for (m, part, cid), g in scored.groupby(["model_key", "part", "config_id"], sort=True):
        w = g.pivot_table(index="item_id", columns="arm", values="correct", aggfunc="first")
        if not set(ARMS) <= set(w.columns):
            continue
        w = w.dropna(subset=list(ARMS))
        e, z = w["eight_shot_cot"].astype(int).to_numpy(), w["zero_shot_cot"].astype(int).to_numpy()
        a, b, c, d = int(((e == 1) & (z == 1)).sum()), int(((e == 1) & (z == 0)).sum()), \
            int(((e == 0) & (z == 1)).sum()), int(((e == 0) & (z == 0)).sum())
        nc = newcombe_paired(a, b, c, d, level)
        n = a + b + c + d
        rows.append({"model_key": m, "part": part, "config_id": cid, "n_items": n, "a": a, "b": b, "c": c, "d": d,
                     "zero_shot": 100 * (a + c) / n if n else math.nan, "eight_shot": 100 * (a + b) / n if n else math.nan,
                     "benefit": 100 * nc["diff"], "benefit_lo": 100 * nc["lo"], "benefit_hi": 100 * nc["hi"],
                     "phi": nc["phi"], "phi_note": nc["phi_note"]})
    return pd.DataFrame(rows)


def sign(x: float) -> int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return 0 if x == 0 else (1 if x > 0 else -1)


def claim_holds(ben: pd.DataFrame, strong: Sequence[str]) -> dict:
    """The published claim under the protocol's rule: no strong model's benefit interval lies above zero."""
    s = ben[ben["model_key"].isin(strong)]
    above = sorted(s.loc[s["benefit_lo"] > 0, "model_key"].tolist())
    return {"holds": len(above) == 0, "strong_models_with_benefit_above_zero": above,
            "strong_models_present": sorted(s["model_key"].tolist())}


def capability_slope(benefit: Sequence[float], zero_shot: Sequence[float]) -> dict:
    """OLS slope of benefit on zero-shot accuracy across models (pp per pp). Undefined with fewer than two models
    or no spread in zero-shot accuracy; a slope of exactly zero is a tie."""
    x, y = np.asarray(zero_shot, float), np.asarray(benefit, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return {"slope": math.nan, "n_models": int(len(x)), "sign": None, "note": "fewer than two models"}
    sx = float(((x - x.mean()) ** 2).sum())
    if sx == 0:
        return {"slope": math.nan, "n_models": int(len(x)), "sign": None, "note": "no spread in zero-shot accuracy"}
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sx)
    return {"slope": slope, "n_models": int(len(x)), "sign": sign(slope), "note": "tie" if slope == 0 else None}


def published_table(cfg: Mapping) -> pd.DataFrame:
    rows = []
    for m, v in cfg["models"].items():
        rows.append({"model_key": m, "group": v["group"], "precision": v.get("precision", "bf16"),
                     "published_zero_shot": float(v["zero_shot"]), "published_eight_shot": float(v["eight_shot"]),
                     "published_benefit": round(float(v["eight_shot"]) - float(v["zero_shot"]), 10),
                     "published_source": v["source"]})
    return pd.DataFrame(rows)


def analyze(scored: pd.DataFrame, cfg: Mapping) -> dict:
    """Part A and Part B summaries under the fixed protocol. Models absent from the scored table are reported as not
    run; nothing is imputed."""
    level = float(cfg.get("interval_level", 0.95))
    pub = published_table(cfg)
    acc = accuracy_table(scored, level)
    ben = benefit_table(scored, level).merge(pub, on="model_key", how="left")
    # A benefit of exactly zero is a tie: neither agreement nor disagreement (None, counted with the undefined).
    ben["sign_agrees"] = [None if sign(b) in (None, 0) else sign(b) == sign(p)
                          for b, p in zip(ben["benefit"], ben["published_benefit"])]
    strong = pub.loc[pub["group"] == "strong", "model_key"].tolist()
    strong_bf16 = pub.loc[(pub["group"] == "strong") & (pub["precision"] == "bf16"), "model_key"].tolist()
    out: dict[str, Any] = {"scorer": SCORER_VERSION, "interval_level": level, "models_planned": pub["model_key"].tolist()}

    a = ben[ben["part"] == "numerical"].copy()
    num_acc = acc[acc["part"] == "numerical"].merge(pub, on="model_key", how="left")
    num_acc["published"] = np.where(num_acc["arm"] == "zero_shot_cot", num_acc["published_zero_shot"],
                                    num_acc["published_eight_shot"])
    num_acc["difference"] = num_acc["accuracy"] - num_acc["published"]
    num_acc["published_inside_interval"] = (num_acc["published"] >= num_acc["lo"]) & (num_acc["published"] <= num_acc["hi"])
    out["numerical"] = {
        "models_run": sorted(a["model_key"].unique().tolist()),
        "models_not_run": sorted(set(pub["model_key"]) - set(a["model_key"])),
        "accuracy": num_acc.to_dict("records"), "benefit": a.to_dict("records"),
        "claim_bf16": claim_holds(a, strong_bf16), "claim_with_fp8": claim_holds(a, strong),
        "sign_agreement": {"agree": int((a["sign_agrees"] == True).sum()),  # noqa: E712
                           "disagree": int((a["sign_agrees"] == False).sum()),  # noqa: E712
                           "tie_or_undefined": int(a["sign_agrees"].isna().sum())},
        "capability_slope": capability_slope(a["benefit"], a["zero_shot"]),
        "published_capability_slope": capability_slope(pub["published_benefit"], pub["published_zero_shot"]),
    }

    t = ben[ben["part"] == "transport"].copy()
    per_cfg = {}
    for cid, g in t.groupby("config_id", sort=True):
        per_cfg[cid] = {"claim_bf16": claim_holds(g, strong_bf16), "claim_with_fp8": claim_holds(g, strong),
                        "sign_agree": int((g["sign_agrees"] == True).sum()),  # noqa: E712
                        "sign_disagree": int((g["sign_agrees"] == False).sum()),  # noqa: E712
                        "capability_slope": capability_slope(g["benefit"], g["zero_shot"]), "models": int(len(g))}
    avg = t.groupby("model_key")[["benefit", "zero_shot"]].mean() if len(t) else pd.DataFrame(columns=["benefit", "zero_shot"])
    complete = t.groupby("model_key")["config_id"].nunique() if len(t) else pd.Series(dtype=int)
    n_cfg = len(per_cfg)
    full = complete[complete == n_cfg].index if n_cfg else []
    out["transport"] = {
        "configurations": sorted(per_cfg), "per_configuration": per_cfg, "benefit": t.to_dict("records"),
        "claim_holds_in": {"bf16": sum(v["claim_bf16"]["holds"] for v in per_cfg.values()),
                           "with_fp8": sum(v["claim_with_fp8"]["holds"] for v in per_cfg.values()), "of": n_cfg},
        "slope_signs": {cid: v["capability_slope"]["sign"] for cid, v in per_cfg.items()},
        "equal_budget_slope": capability_slope(avg.loc[full, "benefit"], avg.loc[full, "zero_shot"]) if len(full) else
        {"slope": math.nan, "n_models": 0, "sign": None, "note": "no model ran every configuration"},
        "models_with_every_configuration": sorted(list(full)),
        "per_model_sign_agreement": {m: {"agree": int((g["sign_agrees"] == True).sum()),  # noqa: E712
                                         "disagree": int((g["sign_agrees"] == False).sum()),  # noqa: E712
                                         "configurations": int(len(g))} for m, g in t.groupby("model_key")},
    }
    out["accuracy_table"] = acc.to_dict("records")
    return out
