r"""Dataset 2 panel by the index rule.

The external index is Epoch's Capabilities Index (ECI). In the analysis, capability is each model's expected
baseline accuracy on the evaluation itself, so the index only decides eligibility, the
selection within families and the model tiers. No GSM8K score of ours is read.

Inputs, all stored in manifests/external/:
    eci_scores_20261030.csv            Epoch's public scores file, fetched 30 October 2026
    hf_metadata_20261030.json          Hugging Face metadata of the mapped checkpoints (revision, size, gating)
    vllm030_arch_facts_20261030.json   vLLM 0.30.0's model registry facts per architecture

Rule:
  1. Every ECI row outside the closed-weights group is mapped by hand to the vendor's own instruction-tuned
     checkpoint on Hugging Face, or excluded with a reason (MAPPING below). Base-only, code-specialised,
     reasoning-native (no documented way to turn reasoning off) and hosted-only entries have no such checkpoint.
  2. A mapped checkpoint is eligible when its repository was created on or before 30 June 2026 (the inventory
     window), its weights are safetensors of at most 72 GB (one 96 GB card, with room for the KV cache), and
     vLLM 0.30.0 supports its architecture.
  3. Families with fewer than three eligible checkpoints are dropped (leave-one-family-out needs families
     with a size range).
  4. Within a family, at most K = 4 checkpoints are kept: the lowest and the highest ECI and, for each of the
     K - 2 evenly spaced interior ECI targets, the nearest checkpoint not yet chosen (ties: lower ECI, then
     fewer parameters, then repository name).
  5. Tiers are ECI terciles of the selected panel.

Extension (all eligible checkpoints):
  with --extension, step 4's cap is removed, so every eligible checkpoint of the kept families is in the panel.
  Steps 1-3 and the inputs are unchanged; tiers are ECI terciles of the extended panel.

PowerShell:
    python scripts/select_panel_d2.py
    python scripts/select_panel_d2.py --extension
Writes manifests/d2_eci_candidates.csv and manifests/d2_panel.json; with --extension, manifests/d2_panel_extension.json
only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "manifests", "external")
ECI_FILE = os.path.join(EXT, "eci_scores_20261030.csv")
HF_FILE = os.path.join(EXT, "hf_metadata_20261030.json")
VLLM_FILE = os.path.join(EXT, "vllm030_arch_facts_20261030.json")
WINDOW_END = "2026-06-30"
MAX_WEIGHTS_GB = 72.0
MIN_PER_FAMILY = 3
K_PER_FAMILY = 4

BASE = "base model only; the vendor released no instruction-tuned checkpoint"
CODE = "code-specialised model"
REASONING = "reasoning-native; no documented way to turn reasoning off"
HOSTED = "hosted service; no open checkpoint"
SIZE = "weights exceed 72 GB"
AFTER = "released after the 30 June 2026 window"
GONE = "vendor repository no longer on Hugging Face"

# ECI name -> (family, Hugging Face repo) for mapped rows, or (family, None, reason) for excluded rows.
MAPPING = {
    "Gemma 2B": ("Gemma", "google/gemma-2b-it"), "Gemma 7B": ("Gemma", "google/gemma-7b-it"),
    "Gemma 2 9B": ("Gemma", "google/gemma-2-9b-it"), "Gemma 2 27B": ("Gemma", "google/gemma-2-27b-it"),
    "Gemma 3 4B": ("Gemma", "google/gemma-3-4b-it"), "Gemma 3 12B": ("Gemma", "google/gemma-3-12b-it"),
    "Gemma 3 27B": ("Gemma", "google/gemma-3-27b-it"), "Gemma 4 26B A4B": ("Gemma", "google/gemma-4-26B-A4B-it"),
    "Gemma 4 31B IT": ("Gemma", "google/gemma-4-31B-it"),
    "Llama 2-7B": ("Llama", "meta-llama/Llama-2-7b-chat-hf"), "Llama 2-13B": ("Llama", "meta-llama/Llama-2-13b-chat-hf"),
    "Llama 3-8B": ("Llama", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "Llama 3.1-8B": ("Llama", "meta-llama/Llama-3.1-8B-Instruct"),
    "Mistral 7B v0.1": ("Mistral", "mistralai/Mistral-7B-Instruct-v0.1"),
    "Mistral 7B v0.3": ("Mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Mistral NeMo": ("Mistral", "mistralai/Mistral-Nemo-Instruct-2407"),
    "Mistral Small 3.1": ("Mistral", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
    "Mixtral 8x7B": ("Mistral", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "Qwen-7B": ("Qwen", "Qwen/Qwen-7B-Chat"), "Qwen-14B": ("Qwen", "Qwen/Qwen-14B-Chat"),
    "Qwen2.5-7B": ("Qwen", "Qwen/Qwen2.5-7B-Instruct"), "Qwen3-8B": ("Qwen", "Qwen/Qwen3-8B"),
    "Qwen3-14B": ("Qwen", "Qwen/Qwen3-14B"), "Qwen3-32B": ("Qwen", "Qwen/Qwen3-32B"),
    "Qwen3-30B-A3B": ("Qwen", "Qwen/Qwen3-30B-A3B"),
    "Qwen3-30B-A3B-Instruct (Jul 2025)": ("Qwen", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    "Qwen3.5-9B": ("Qwen", "Qwen/Qwen3.5-9B"), "Qwen3.5-35B-A3B": ("Qwen", "Qwen/Qwen3.5-35B-A3B"),
    "Qwen 3.6 35B-A3B": ("Qwen", "Qwen/Qwen3.6-35B-A3B"), "Qwen3.6 27B": ("Qwen", "Qwen/Qwen3.6-27B"),
    "phi-3-mini 3.8B": ("Phi", "microsoft/Phi-3-mini-4k-instruct"),
    "phi-3-small 7.4B": ("Phi", "microsoft/Phi-3-small-8k-instruct"),
    "phi-3-medium 14B": ("Phi", "microsoft/Phi-3-medium-4k-instruct"), "Phi-4": ("Phi", "microsoft/phi-4"),
    "Falcon-7B": ("Falcon", "tiiuae/falcon-7b-instruct"), "Yi 6B": ("Yi", "01-ai/Yi-6B-Chat"),
    "Yi-34B": ("Yi", "01-ai/Yi-34B-Chat"),
    "Baichuan 2-7B": ("Baichuan", "baichuan-inc/Baichuan2-7B-Chat"),
    "Baichuan2-13B": ("Baichuan", "baichuan-inc/Baichuan2-13B-Chat"),
    "INTELLECT-1": ("INTELLECT", "PrimeIntellect/INTELLECT-1-Instruct"),
    "XGen-7B": ("XGen", "Salesforce/xgen-7b-8k-inst"),
    "Falcon-40B": ("Falcon", "tiiuae/falcon-40b-instruct"),
    # excluded at mapping
    "MPT-7B": ("MPT", None, GONE + " (mosaicml/mpt-7b-instruct)"),
    "MPT-30B": ("MPT", None, GONE + " (mosaicml/mpt-30b-instruct)"),
    "Dolly 2.0-12b": ("Dolly", None, GONE + " (databricks/dolly-v2-12b)"),
    "LLaMA-7B": ("Llama", None, BASE), "LLaMA-13B": ("Llama", None, BASE), "LLaMA-33B": ("Llama", None, BASE),
    "LLaMA-65B": ("Llama", None, BASE), "Cerebras-GPT-13B": ("Cerebras-GPT", None, BASE),
    "Baichuan1-7B": ("Baichuan", None, BASE), "Phi-1.5": ("Phi", None, BASE), "Phi-2": ("Phi", None, BASE),
    "Falcon 2 11B": ("Falcon", None, BASE),
    "DeepSeek Coder 1.3B": ("DeepSeek", None, CODE), "DeepSeek Coder 6.7B": ("DeepSeek", None, CODE),
    "DeepSeek Coder 33B": ("DeepSeek", None, CODE), "StarCoder 2 3B": ("StarCoder", None, CODE),
    "StarCoder 2 7B": ("StarCoder", None, CODE), "StarCoder 2 15B": ("StarCoder", None, CODE),
    "Qwen2.5-Coder (1.5B)": ("Qwen", None, CODE), "Qwen2.5-Coder (7B)": ("Qwen", None, CODE),
    "Qwen2.5-Coder-32B": ("Qwen", None, CODE), "Kimi K2.7 Code": ("Kimi", None, CODE),
    "DeepSeek-R1": ("DeepSeek", None, REASONING), "DeepSeek-R1 (May 2025)": ("DeepSeek", None, REASONING),
    "QwQ-32B": ("Qwen", None, REASONING), "Qwen3-30B-A3B-Thinking (Jul 2025)": ("Qwen", None, REASONING),
    "Qwen3-235B-A22B-Thinking (Jul 2025)": ("Qwen", None, REASONING),
    "Magistral Small 1.0": ("Mistral", None, REASONING), "gpt-oss-20b": ("gpt-oss", None, REASONING),
    "gpt-oss-120b": ("gpt-oss", None, REASONING), "Kimi K2 Thinking": ("Kimi", None, REASONING),
    "Qwen 3.5 Plus (hosted 397B-A17B)": ("Qwen", None, HOSTED),
    "Qwen 3.5 Flash (hosted 35B-A3B)": ("Qwen", None, HOSTED), "Qwen 3.6 Max (Preview)": ("Qwen", None, HOSTED),
    "Qwen 3.6 Flash": ("Qwen", None, HOSTED),
    "Llama 2-70B": ("Llama", None, SIZE), "Llama 3-70B": ("Llama", None, SIZE),
    "Llama 3.1-70B": ("Llama", None, SIZE), "Llama 3.1-405B": ("Llama", None, SIZE),
    "Llama 3.2 90B": ("Llama", None, SIZE), "Llama 3.3 70B": ("Llama", None, SIZE),
    "Llama 4 Scout": ("Llama", None, SIZE), "Llama 4 Maverick": ("Llama", None, SIZE),
    "Stable Beluga 2": ("Stable Beluga", None, SIZE), "Falcon-180B": ("Falcon", None, SIZE),
    "Mixtral 8x22B": ("Mistral", None, SIZE), "Mistral Large 2 (Jul 2024)": ("Mistral", None, SIZE),
    "Mistral Large 2 (Nov 2024)": ("Mistral", None, SIZE),
    "Mistral Medium 3.5": ("Mistral", None, SIZE + " (mistralai/Mistral-Medium-3.5-128B)"),
    "Command R+": ("Command", None, SIZE), "Qwen2-72B": ("Qwen", None, SIZE), "Qwen2.5-72B": ("Qwen", None, SIZE),
    "Qwen3-235B-A22B": ("Qwen", None, SIZE), "Qwen3-235B-A22B-Instruct (Jul 2025)": ("Qwen", None, SIZE),
    "Qwen3.5 397B-A17B": ("Qwen", None, SIZE),
    "DeepSeek-V2 (MoE-236B, May 2024)": ("DeepSeek", None, SIZE), "DeepSeek-V3": ("DeepSeek", None, SIZE),
    "DeepSeek-V3 (Mar 2025)": ("DeepSeek", None, SIZE), "DeepSeek-V3.1": ("DeepSeek", None, SIZE),
    "DeepSeek-V3.2-Exp": ("DeepSeek", None, SIZE), "DeepSeek-V3.2": ("DeepSeek", None, SIZE),
    "DeepSeek-V4-Pro": ("DeepSeek", None, SIZE), "DeepSeek-V4-Flash": ("DeepSeek", None, SIZE),
    "Kimi K2 (Jul 2025)": ("Kimi", None, SIZE), "Kimi K2.5": ("Kimi", None, SIZE), "Kimi K2.6": ("Kimi", None, SIZE),
    "GLM-4.7": ("GLM", None, SIZE), "GLM-5": ("GLM", None, SIZE), "GLM-5.1": ("GLM", None, SIZE),
    "GLM-5.2": ("GLM", None, SIZE), "MiniMax-M2.5": ("MiniMax", None, SIZE), "MiniMax-M2.7": ("MiniMax", None, SIZE),
    "MiniMax-M3": ("MiniMax", None, SIZE), "Nemotron 3 Ultra": ("Nemotron", None, SIZE),
    "Inkling-Small": ("Inkling", None, AFTER), "Inkling": ("Inkling", None, AFTER), "Kimi K3": ("Kimi", None, AFTER),
    "DeepSeek V4 Flash 0731": ("DeepSeek", None, AFTER), "DeepSeek V4 Pro 0813": ("DeepSeek", None, AFTER),
    "GLM-5.3": ("GLM", None, AFTER), "Qwen 3.8 27B": ("Qwen", None, AFTER), "GLM-5.3-Flash": ("GLM", None, AFTER),
    "DeepSeek V4.1 Flash": ("DeepSeek", None, AFTER), "Gemini 4 Argon": ("Gemini", None, HOSTED),
    "Beam 1.0": ("Beam", None, AFTER), "Mistral Large 4": ("Mistral", None, AFTER),
    "Muse Spark Open": ("Llama", None, AFTER), "DeepSeek V4.1 Flash 1017": ("DeepSeek", None, AFTER),
}
# Metadata read on 30 October 2026 for mapped repositories missing from hf_metadata_20261030.json.
EXTRA_HF = {
    "INTELLECT-1": {"repo": "PrimeIntellect/INTELLECT-1-Instruct", "sha": "14d9716e61324759fffd57b91d13db19765ca1c0",
                    "created": "2024-11-28", "gated": False, "params_b": 10.21, "weights_gb": 20.4,
                    "arch": ["LlamaForCausalLM"]},
    "XGen-7B": {"repo": "Salesforce/xgen-7b-8k-inst", "sha": "17c7af360a95ba2014ad77f146156846235f803f",
                "created": "2023-06-28", "gated": False, "params_b": None, "weights_gb": None,
                "arch": ["LlamaForCausalLM"]},
    "Falcon-40B": {"repo": "tiiuae/falcon-40b-instruct", "sha": "ecb78d97ac356d098e79f0db222c9ce7c5d9ee5f",
                   "created": "2023-05-25", "gated": False, "params_b": None, "weights_gb": None,
                   "arch": ["FalconForCausalLM"]},
}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def candidates(eci: pd.DataFrame, hf: dict, archs: dict) -> pd.DataFrame:
    """One row per non-closed ECI entry, with its mapping, metadata and eligibility verdict."""
    rows = []
    for _, e in eci[eci["Accessibility group"] != "Closed weights"].iterrows():
        name = e["Model"]
        if name not in MAPPING:
            raise SystemExit(f"ECI row {name!r} has no mapping decision")
        m = MAPPING[name]
        r = {"eci_name": name, "eci": float(e["eci"]), "eci_ci_low": float(e["eci_ci_low"]),
             "eci_ci_high": float(e["eci_ci_high"]), "eci_date": e["date"], "family": m[0], "repo": m[1]}
        if m[1] is None:
            r.update(eligible=False, reason=m[2])
            rows.append(r)
            continue
        meta = hf.get(name) or EXTRA_HF.get(name)
        if meta is None or meta.get("error"):
            raise SystemExit(f"{name}: no Hugging Face metadata ({meta})")
        arch = (meta.get("arch") or [None])[0]
        fact = archs.get(arch, {})
        r.update(revision=meta["sha"], created=meta["created"], gated=meta["gated"], license=meta.get("license"),
                 params_b=meta.get("params_b"), weights_gb=meta.get("weights_gb"), architecture=arch,
                 vllm_supported=bool(fact.get("supported")), vllm_hybrid=fact.get("is_hybrid"))
        if r["created"] > WINDOW_END:
            reason = AFTER
        elif r["weights_gb"] is None:
            reason = "no safetensors weights"
        elif r["weights_gb"] > MAX_WEIGHTS_GB:
            reason = SIZE
        elif not r["vllm_supported"]:
            reason = f"architecture {arch} not supported by vLLM 0.30.0"
        else:
            reason = ""
        r.update(eligible=reason == "", reason=reason)
        rows.append(r)
    return pd.DataFrame(rows)


def select_family(fam: pd.DataFrame, k: int | None = K_PER_FAMILY) -> pd.DataFrame:
    """Lowest and highest ECI, then the nearest checkpoint to each evenly spaced interior target. k=None keeps every
    eligible checkpoint (the extension)."""
    fam = fam.sort_values(["eci", "params_b", "repo"]).reset_index(drop=True)
    if k is None or len(fam) <= k:
        return fam.assign(selected_for="all eligible")
    lo, hi = fam["eci"].iloc[0], fam["eci"].iloc[-1]
    chosen = {0: "lowest ECI", len(fam) - 1: "highest ECI"}
    for j in range(1, k - 1):
        target = lo + j * (hi - lo) / (k - 1)
        free = fam.drop(index=list(chosen))
        dist = (free["eci"] - target).abs()
        best = free.assign(d=dist).sort_values(["d", "eci", "params_b", "repo"]).index[0]
        chosen[best] = f"nearest to interior target {target:.2f}"
    out = fam.loc[sorted(chosen)].copy()
    out["selected_for"] = [chosen[i] for i in sorted(chosen)]
    return out


def select_panel(cand: pd.DataFrame, k: int | None = K_PER_FAMILY) -> tuple[pd.DataFrame, dict]:
    elig = cand[cand["eligible"]]
    counts = elig.groupby("family").size().to_dict()
    kept = [f for f, n in counts.items() if n >= MIN_PER_FAMILY]
    panel = pd.concat([select_family(elig[elig["family"] == f], k) for f in sorted(kept)], ignore_index=True)
    q = np.quantile(panel["eci"], [1 / 3, 2 / 3])
    panel["tier"] = np.where(panel["eci"] <= q[0], "low", np.where(panel["eci"] <= q[1], "middle", "high"))
    families = {"eligible_per_family": counts, "kept": sorted(kept),
                "dropped": sorted(f for f in counts if f not in kept), "tier_cutpoints_eci": [float(x) for x in q]}
    return panel, families


def main(argv=None):
    extension = "--extension" in (sys.argv[1:] if argv is None else argv)
    eci = pd.read_csv(ECI_FILE)
    hf = json.load(open(HF_FILE, encoding="utf-8"))
    archs = json.load(open(VLLM_FILE, encoding="utf-8"))["archs"]
    cand = candidates(eci, hf, archs)
    if not extension:
        cand.to_csv(os.path.join(ROOT, "manifests", "d2_eci_candidates.csv"), index=False)
    panel, families = select_panel(cand, None if extension else K_PER_FAMILY)
    cols = ["family", "eci_name", "repo", "revision", "eci", "eci_ci_low", "eci_ci_high", "tier", "selected_for",
            "params_b", "weights_gb", "architecture", "vllm_hybrid", "gated", "license", "created"]
    rule = __doc__.split("Rule:")[1].split("PowerShell:")[0].strip()
    out = {
        "rule": rule if extension else rule.split("Extension (")[0].strip(),
        "decided": ("index rule with the per-family cap removed" if extension else
                    "index rule with the per-family cap"),
        "inputs": {os.path.relpath(p, ROOT).replace("\\", "/"): sha256(p) for p in (ECI_FILE, HF_FILE, VLLM_FILE)},
        "parameters": {"window_end": WINDOW_END, "max_weights_gb": MAX_WEIGHTS_GB,
                       "min_eligible_per_family": MIN_PER_FAMILY, "k_per_family": None if extension else K_PER_FAMILY},
        "counts": {"eci_rows_not_closed": int(len(cand)), "mapped": int(cand["repo"].notna().sum()),
                   "eligible": int(cand["eligible"].sum()), "panel": int(len(panel))},
        "families": families,
        "panel": json.loads(panel[cols].to_json(orient="records")),
    }
    if extension:
        base = {p["repo"] for p in json.load(open(os.path.join(ROOT, "manifests", "d2_panel.json"), encoding="utf-8"))["panel"]}
        out["main_panel"] = "manifests/d2_panel.json"
        out["added_to_main"] = sorted(r for r in panel["repo"] if r not in base)
    name = "d2_panel_extension.json" if extension else "d2_panel.json"
    with open(os.path.join(ROOT, "manifests", name), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out["counts"]), json.dumps(families))
    print(panel[["family", "repo", "eci", "tier", "selected_for", "weights_gb", "vllm_hybrid", "gated"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
