r"""Does w rise with release date, and intervals for the newest-model reversal shares (\aeRecent<T>ReversedPct).

From the repository root, in this order:
    $env:PYTHONIOENCODING = "utf-8"
    .\.venv\Scripts\python.exe scripts\release_dates.py dates   # manifests/panel_release_dates.json (no w read)
    .\.venv\Scripts\python.exe scripts\release_dates.py pairs   # manifests/panel_release_pairs.json (no w read)
    .\.venv\Scripts\python.exe scripts\release_dates.py run     # results/release_dates/ (reads w)

dates  Every checkpoint's release date with its source. The 29 panel models: the Hugging Face repository creation date
       ('created' in manifests/d2_panel_extension.json, from manifests/external/hf_metadata_20261030.json; model key to
       repository by the model registries), read through scripts/statement_values.release_dates, the function that set
       \aeRecentM, so the 14 checkpoints from 2025 on are the same 14. The seven of the extension: the
       repository creation date where the repository records one (manifests/dataset2_inventory.md, whose "Released"
       column is the repository creation date), else the date the developer's page or the provider's listing gives,
       each with its source; recorded alternatives beside them. Write-once: an identical file is kept, a different
       one refused (--force overwrites).
pairs  The within-family pairing rule, from names, sizes and dates only, written with the time it was
       written, before any w is read by this script. Write-once as above.
run    Reads w (the share of requested replies over 16 tokens, discovery draws, src/baseline_mixture.RULE_TEXT):
       panel from results/baseline_mixture/baseline_mixture.json (w_discovery), checked against
       results/frontier_panel/closed_w.json's panel block; the six ladder models from results/ladder/analysis/
       ladder.json (cut reading; w is the same in both readings); Qwen4-Max from closed_w.json (API count, discovery
       draws, 100 items per benchmark). (a) Spearman rho of w with release date, 95% percentile interval from
       resampling whole families; (b) mean w of the checkpoints from 2025 on and of the earlier ones; (c) the pairs of
       the manifest, newer against older; (d) the extension, descriptive. Then the reversed share of model pairs
       (CoT - requested against CoT - forced, strict, model means over the discovery draws) among the checkpoints from
       2025 on, with a model bootstrap and a family bootstrap (src/decision_directions.reversal_uncertainty); the point
       values must equal the stored \aeRecent<T> counts (results/statement_values/statement_values.json) or the script stops.

Outputs: results/release_dates/release_dates.json (block "macros"), release_dates.md, release_dates_clean.md.
"""
from __future__ import annotations

import argparse
import datetime as dtm
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQUEST = "release dates and newest-model reversal shares"
DATES = os.path.join(ROOT, "manifests", "panel_release_dates.json")
PAIRS = os.path.join(ROOT, "manifests", "panel_release_pairs.json")
OUT = os.path.join(ROOT, "results", "release_dates")
STEM = "release_dates"
TASKS = ("gsm8k", "math", "bbh")
T_TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
T_NAME = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard"}
RECENT_FROM = "2025-01-01"          # as scripts/statement_values.RECENT_FROM (checked at run time)
N_BOOT = 9999                       # as scripts/run_decision_directions.N_BOOT (checked at run time)
TOL = 1e-9
SIZE_RATIO = 1.5                    # the pairing rule's size band (larger total / smaller total)
SIZE_RATIO_ALT = 2.0                # the same rule with a wider band, written at the same time, summary only
LADDER6 = ("ministral-3-8b-instruct-2512", "ministral-3-14b-instruct-2512", "mistral-large-2512",
           "llama-3.1-70b-instruct-fp8", "qwen3.5-122b-a10b-fp8", "deepseek-v4.1-flash-1017")
CLOSED = "qwen4-max"
INVENTORY = "manifests/dataset2_inventory.md"
INVENTORY_CONVENTION = "The release column is the repository creation date (UTC)"
MOE_TAG = re.compile(r"-a(\d+(?:\.\d+)?)b(?:-|$)", re.I)

# w values printed while the files' layout was read in this session, before the pairing rule was written (as they
# were shown, to the precision shown). Recorded so the reader knows what was seen; the rule uses no w.
SEEN_BEFORE_RULE = [
    "results/baseline_mixture/baseline_mixture.json, the first three models of each benchmark's list: GSM8K "
    "gemma-4-31b-it 0.995, meta-llama-3-8b-instruct 0.310, mistral-7b-instruct-v0.3 0.482; MATH-500 gemma-4-31b-it "
    "0.998, meta-llama-3-8b-instruct 0.108, mistral-7b-instruct-v0.3 0.451; BIG-Bench Hard meta-llama-3-8b-instruct "
    "0.060, mistral-7b-instruct-v0.3 0.197, mistral-7b-instruct-v0.1 0.369",
    "results/frontier_panel/closed_w.json and closed_w.md: gemma-2-27b-it GSM8K 0.014, deepseek-v4.1-flash-1017 "
    "GSM8K 0.130, Qwen4-Max GSM8K 76.2%, MATH-500 75.7%, BIG-Bench Hard 38.8%, HotpotQA 0.9%",
]


def rel(p) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else "nan")
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def num(x, nd):
    import macros_added as ma
    return ma.num(x, nd)


def write_once(path: str, obj: dict, compare_keys: tuple, force: bool) -> str:
    """Write obj as JSON; an existing file equal on compare_keys is kept as it is, a different one refused."""
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        if all(old.get(k) == jsonable(obj).get(k) for k in compare_keys):
            return "kept (identical)"
        diff = [k for k in compare_keys if old.get(k) != jsonable(obj).get(k)]
        raise SystemExit(f"{rel(path)} exists and differs in {diff}; it is write-once (--force overwrites)")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(obj), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return "written"


# ============================================================================ dates


def inventory_date(repo: str) -> str:
    """The date in the inventory's table row of `repo` (exactly one row, exactly one date in it)."""
    with open(os.path.join(ROOT, INVENTORY), encoding="utf-8") as fh:
        txt = fh.read()
    if INVENTORY_CONVENTION not in txt:
        raise SystemExit(f"{INVENTORY}: the convention sentence ({INVENTORY_CONVENTION!r}) is gone")
    rows = [ln for ln in txt.splitlines() if ln.startswith("|") and f"`{repo}`" in ln]
    if len(rows) != 1:
        raise SystemExit(f"{INVENTORY}: {len(rows)} table rows name `{repo}`")
    ds = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", rows[0])
    if len(ds) != 1:
        raise SystemExit(f"{INVENTORY}: the row of `{repo}` holds {len(ds)} dates")
    return ds[0]


def inventory_sentence_date(repo: str) -> str:
    """A date written right after `repo` in the inventory's running text: `repo` (YYYY-MM-DD)."""
    with open(os.path.join(ROOT, INVENTORY), encoding="utf-8") as fh:
        txt = fh.read()
    m = re.findall(r"`" + re.escape(repo) + r"` \((20\d\d-\d\d-\d\d)\)", txt)
    if len(set(m)) != 1:
        raise SystemExit(f"{INVENTORY}: {len(set(m))} dates written after `{repo}`")
    return m[0]


def find_key(d, key: str):
    """The value under `key` anywhere in a nested dict (exactly one)."""
    hits = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key:
                    hits.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(d)
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} entries named {key!r}")
    return hits[0]


def panel_keys() -> dict:
    """{task: [model keys]} of the panel as the w file lists them (models only; no w is read here)."""
    B = load_json("results", "baseline_mixture", "baseline_mixture.json")["tasks"]
    return {t: list(B[t]["models"]) for t in TASKS}


def build_dates() -> dict:
    import statement_values as wb
    if wb.RECENT_FROM != RECENT_FROM:
        raise SystemExit("scripts/statement_values.RECENT_FROM differs from this script's")
    pk = panel_keys()
    keys = sorted(set().union(*pk.values()))
    d, missing = wb.release_dates(keys)
    if missing:
        raise SystemExit(f"no recorded date for {missing}")
    man = {p["repo"]: p for p in load_json("manifests", "d2_panel_extension.json")["panel"]}
    entries = {}
    for k in keys:
        repo = d[k]["repo"]
        p = man[repo]
        entries[k] = {"group": "panel", "family": p["family"], "repo": repo, "params_b": p["params_b"],
                      "date": d[k]["created"], "basis": "hf_repo_created",
                      "source": "manifests/d2_panel_extension.json 'created' (from manifests/external/"
                                "hf_metadata_20261030.json, the Hugging Face API's repository creation date, read 30 "
                                "Oct 2026), through scripts/statement_values.release_dates",
                      "hf_metadata_agrees": d[k]["agrees_with_hf_metadata"],
                      "benchmarks": [t for t in TASKS if k in pk[t]]}
    # the 14 of \aeRecentM, and each benchmark's count, as stored
    W = load_json("results", "statement_values", "statement_values.json")["recent"]
    recent = sorted(k for k, e in entries.items() if e["date"] >= RECENT_FROM)
    if recent != sorted(W["union_models"]):
        raise SystemExit("the checkpoints from 2025 on differ from results/statement_values/statement_values.json's 14")
    for t in TASKS:
        if sorted(k for k in pk[t] if entries[k]["date"] >= RECENT_FROM) != sorted(W["tasks"][t]["models"]):
            raise SystemExit(f"{t}: the recent models differ from statement_values.json's")

    # the extension
    J = load_json("results", "ladder", "analysis", "ladder.json")
    lfam = {k: J["models"][k]["family"] for k in LADDER6}
    llama8 = entries["llama-3.1-8b-instruct"]
    if inventory_date("meta-llama/Llama-3.1-8B-Instruct") != llama8["date"]:
        raise SystemExit("the inventory's Llama-3.1-8B-Instruct date differs from the panel manifest's")
    mp = load_json("results", "ladder", "mistral_price.json")["model"]
    if mp["api_id"] != "mistral-large-2512":
        raise SystemExit("results/ladder/mistral_price.json is not Mistral Large 3's record")
    ps = load_json("manifests", "ladder_published_sizes.json")["models"]["deepseek-v4.1-flash-1017"]
    m = re.search(r"news(\d{6}), (\d{1,2} [A-Z][a-z]{2} \d{4})", ps["checkpoint"])
    if not m:
        raise SystemExit("manifests/ladder_published_sizes.json: DeepSeek 1017's release-note date not found")
    ds_date = dtm.datetime.strptime(m.group(2), "%d %b %Y").date().isoformat()
    if ds_date[2:].replace("-", "") != m.group(1):
        raise SystemExit("DeepSeek 1017's release-note URL and date disagree")
    ql = load_json("results", "frontier_panel", "closed_ladder_price.json")["openrouter"]["models"]["qwen/qwen4-max"]
    slug = re.search(r"-(\d{8})$", ql["canonical_slug"]).group(1)
    mistral3_public = mp["released"]
    ext = {
        "ministral-3-8b-instruct-2512": {
            "repo": "mistralai/Ministral-3-8B-Instruct-2512", "date": inventory_date("mistralai/Ministral-3-8B-Instruct-2512"),
            "basis": "hf_repo_created", "source": f"{INVENTORY}, Table 2 (repository creation date)",
            "alternatives": [{"date": mistral3_public, "what": "Mistral 3's public release: Mistral Large 3's "
                              "release date on Mistral's model page (results/ladder/mistral_price.json, 'released'); "
                              "the Ministral 3 checkpoints carry the same 2512 tag"}]},
        "ministral-3-14b-instruct-2512": {
            "repo": "mistralai/Ministral-3-14B-Instruct-2512",
            "date": inventory_date("mistralai/Ministral-3-14B-Instruct-2512"),
            "basis": "hf_repo_created", "source": f"{INVENTORY}, Table 2 (repository creation date)",
            "alternatives": [{"date": mistral3_public, "what": "as for Ministral 3 8B"}]},
        "mistral-large-2512": {
            "repo": "mistralai/Mistral-Large-3-675B-Instruct-2512 (run through Mistral's API)", "date": mistral3_public,
            "basis": "developer_page_released",
            "source": f"results/ladder/mistral_price.json, 'released' (Mistral's model page {mp['page']}, read "
                      f"{load_json('results', 'ladder', 'mistral_price.json')['date'][:10]}); no repository creation "
                      "date is recorded in this repository",
            "alternatives": []},
        "llama-3.1-70b-instruct-fp8": {
            "repo": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8-dynamic (Meta's meta-llama/Llama-3.1-70B-Instruct, "
                    "quantized)", "date": llama8["date"], "basis": "same_release_as_panel_checkpoint",
            "source": "the Llama 3.1 release: the repository creation date of the panel's meta-llama/Llama-3.1-8B-"
                      "Instruct (same source as the panel; equal to the inventory's row); the creation date of "
                      "meta-llama/Llama-3.1-70B-Instruct is not recorded in this repository",
            "alternatives": []},
        "qwen3.5-122b-a10b-fp8": {
            "repo": "Qwen/Qwen3.5-122B-A10B (run as Qwen/Qwen3.5-122B-A10B-FP8)",
            "date": inventory_date("Qwen/Qwen3.5-122B-A10B"), "basis": "hf_repo_created",
            "source": f"{INVENTORY}, Table 1 (repository creation date of the bfloat16 release)",
            "alternatives": []},
        "deepseek-v4.1-flash-1017": {
            "repo": "deepseek-ai/DeepSeek-V4.1-Flash-1017 (run through Nebius)", "date": ds_date,
            "basis": "developer_release_note",
            "source": "manifests/ladder_published_sizes.json (DeepSeek's release note of the 1017 checkpoint, "
                      "https://api-docs.deepseek.com/news/news" + m.group(1) + "); no repository creation date of "
                      "the 1017 checkpoint is recorded in this repository",
            "alternatives": [{"date": inventory_sentence_date("deepseek-ai/DeepSeek-V4.1-Flash"),
                              "what": f"deepseek-ai/DeepSeek-V4.1-Flash, the V4.1-Flash release the 1017 checkpoint "
                                      f"updates ({INVENTORY}, repository creation date)"}]},
        CLOSED: {
            "repo": "none (closed weights; qwen/qwen4-max on OpenRouter)", "date": ql["created"],
            "basis": "provider_listing_created",
            "source": "results/frontier_panel/closed_ladder_price.json, OpenRouter's model listing, 'created' "
                      f"(canonical slug {ql['canonical_slug']})",
            "alternatives": [{"date": f"{slug[:4]}-{slug[4:6]}-{slug[6:]}", "what": "the date in OpenRouter's "
                              "canonical slug"}]},
    }
    for k, e in ext.items():
        e.update({"group": "extension", "family": lfam.get(k, "Qwen" if k == CLOSED else None)})
        entries[k] = e
    flags = [
        {"model": k, "what": "repository created about a month before the public release recorded beside it "
                             f"({entries[k]['date']} against {mistral3_public}); both on the same side of "
                             f"{RECENT_FROM}", "from": "recorded sources"}
        for k in ("ministral-3-8b-instruct-2512", "ministral-3-14b-instruct-2512")]
    flags += [
        {"model": "phi-4", "what": "repository created 2024-12-11; Microsoft announced Phi-4 on 12 December 2024 and, "
                                  "as recalled, opened the weights on Hugging Face in early January 2025, so it sits "
                                  "just before the 2025 cut by its repository date and after it by its open-weights "
                                  "date", "from": "general knowledge, not read from a source in this run"},
        {"model": "gemma-3-4b-it", "what": "repository created 2025-02-20; Gemma 3 was announced on 12 March 2025 "
                                           "(the 12B and 27B repositories were created 2025-03-01)",
         "from": "general knowledge, not read from a source in this run"},
    ]
    return {"what": "Release dates of the panel's 29 checkpoints and of the seven of the extension; "
                    "written before any w is read by scripts/release_dates.py",
            "request": REQUEST, "script": "scripts/release_dates.py dates",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "panel_rule": "Hugging Face repository creation date (UTC), the source of \\aeRecentM; a checkpoint is "
                          f"recent when its date is on or after {RECENT_FROM}",
            "extension_rule": "the repository creation date where this repository records one; otherwise the "
                              "developer's or the provider's recorded date, its basis named; alternatives listed",
            "recent_from": RECENT_FROM, "recent_panel": recent, "recent_panel_M": len(recent),
            "entries": entries, "flags": flags}


# ============================================================================ pairs


def size_of(key: str, e: dict, ladder_sizes: dict) -> dict | None:
    """{'total', 'active', 'moe'} from the recorded sizes (panel: params_b and the name's A<n>B tag; extension:
    ladder.json's sizes block); None when no size is recorded."""
    if e["group"] == "panel":
        m = MOE_TAG.search(key)
        tot = float(e["params_b"])
        return {"total": tot, "active": float(m.group(1)) if m else tot, "moe": bool(m),
                "source": "params_b of manifests/d2_panel_extension.json; active parameters from the name's A<n>B tag"}
    s = ladder_sizes.get(key)
    if not s or s.get("total") is None:
        return None
    tot = float(s["total"])
    moe = s.get("arch") == "mixture of experts"
    act = s.get("active")
    return {"total": tot, "active": float(act) if act is not None else None, "moe": moe,
            "source": "results/ladder/analysis/ladder.json, sizes"}


def similar(a: dict, b: dict, ratio: float) -> bool:
    if a is None or b is None or a["moe"] != b["moe"]:
        return False
    if max(a["total"], b["total"]) / min(a["total"], b["total"]) > ratio:
        return False
    if a["moe"]:
        if a["active"] is None or b["active"] is None:
            return False
        return max(a["active"], b["active"]) / min(a["active"], b["active"]) <= ratio
    return True


def pair_rule(keys: list[str], entries: dict, sizes: dict, ratio: float) -> list[dict]:
    """For each checkpoint (the newer), the most recent checkpoint of the same family released strictly earlier and of
    similar size (same architecture kind; total parameters, and for mixtures of experts also active parameters,
    within a factor of `ratio`); a tie in date goes to the nearer total size (|log ratio|), then to the key in
    alphabetical order. At most one pair per newer checkpoint."""
    out = []
    for n in sorted(keys):
        en, sn = entries[n], sizes.get(n)
        cand = [o for o in keys if o != n and entries[o]["family"] == en["family"] and en["family"] is not None
                and entries[o]["date"] < en["date"] and similar(sizes.get(o), sn, ratio)]
        if not cand:
            continue
        best = sorted(cand, key=lambda o: (-dtm.date.fromisoformat(entries[o]["date"]).toordinal(),
                                           abs(math.log(sizes[o]["total"] / sn["total"])), o))[0]
        out.append({"older": best, "newer": n, "family": en["family"], "older_date": entries[best]["date"],
                    "newer_date": en["date"], "older_total_b": sizes[best]["total"], "newer_total_b": sn["total"],
                    "moe": sn["moe"], "candidates": len(cand)})
    return out


PAIR_RULE_TEXT = (
    "Within a family (the panel manifest's family; the ladder's for the extension), each checkpoint is paired with the "
    "most recent checkpoint of the same family released strictly earlier (release dates of manifests/"
    "panel_release_dates.json) and of similar size. Similar size: the same architecture kind (dense, or mixture of "
    "experts as the name's A<n>B tag or the recorded size says) and total parameters within a factor of {r} (larger "
    "over smaller); for mixtures of experts the active parameters also within that factor. A tie in date goes to the "
    "nearer total size, then to the key in alphabetical order. Each checkpoint is the newer side of at most one pair; "
    "a checkpoint can be the older side of several. Checkpoints released on the same date are not paired. Sizes: "
    "params_b of manifests/d2_panel_extension.json (panel), the sizes block of results/ladder/analysis/ladder.json "
    "(extension); a checkpoint without a recorded size (Qwen4-Max) enters no pair. Counted on a benchmark: the pairs "
    "whose two checkpoints both have w there; the newer is higher when its w exceeds the older's by more than 1e-9, "
    "a tie when the two are within 1e-9.")


def build_pairs(D: dict) -> dict:
    entries = D["entries"]
    J = load_json("results", "ladder", "analysis", "ladder.json")
    sizes = {k: size_of(k, e, J["sizes"]) for k, e in entries.items()}
    panel = [k for k, e in entries.items() if e["group"] == "panel"]
    allk = list(entries)
    P = pair_rule(panel, entries, sizes, SIZE_RATIO)
    PA = pair_rule(panel, entries, sizes, SIZE_RATIO_ALT)
    E = pair_rule(allk, entries, sizes, SIZE_RATIO)
    pset = {(p["older"], p["newer"]) for p in P}
    ext_pairs = [p for p in E if (p["older"], p["newer"]) not in pset]
    if not pset <= {(p["older"], p["newer"]) for p in E}:
        raise SystemExit("adding the extension changes a panel pair")
    return {"what": "Within-family pairs of newer and older releases from names, "
                    "sizes and dates only, written before any w is read by scripts/release_dates.py",
            "request": REQUEST, "script": "scripts/release_dates.py pairs",
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "dates_manifest": rel(DATES), "dates_written_at": D["written_at"],
            "rule": PAIR_RULE_TEXT.format(r=SIZE_RATIO), "size_ratio": SIZE_RATIO,
            "rule_alt": PAIR_RULE_TEXT.format(r=SIZE_RATIO_ALT) + " (A wider size band, written with the main rule "
                        "and reported in the summary only.)", "size_ratio_alt": SIZE_RATIO_ALT,
            "seen_before_rule": SEEN_BEFORE_RULE,
            "seen_note": "These w values were printed while the files' layout was checked, before this rule was "
                         "written. The rule was set from the names, sizes and dates listed here; no other w was read.",
            "sizes": sizes, "pairs": P, "pairs_alt": PA, "extension_pairs": ext_pairs}


# ============================================================================ run


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def pct_ci(v) -> list | None:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None


def family_bootstrap_rho(x, y, fams, n_boot: int, rng) -> dict:
    """Spearman rho over the models of whole families drawn with replacement (a family drawn twice enters twice)."""
    groups: dict = {}
    for j, f in enumerate(fams):
        groups.setdefault(f, []).append(j)
    keys = sorted(groups)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([groups[keys[p]] for p in pick])
        out[b] = spearman(x[ix], y[ix])
    lofo = {f: spearman(x[[j for j in range(len(x)) if fams[j] != f]], y[[j for j in range(len(x)) if fams[j] != f]])
            for f in keys}
    return {"n_boot": int(n_boot), "n_families": len(keys), "ci95": pct_ci(out),
            "n_undefined": int((~np.isfinite(out)).sum()), "leave_one_family_out": lofo}


def ordinal(d: str) -> int:
    return dtm.date.fromisoformat(d).toordinal()


def load_w() -> tuple[dict, list[str]]:
    """{task: {model: w}} for the panel, the six ladder models and Qwen4-Max; checks."""
    B = load_json("results", "baseline_mixture", "baseline_mixture.json")
    from src import baseline_mixture as bm
    if B["rule"] != bm.RULE_TEXT:
        raise SystemExit("baseline_mixture.json's w rule differs from src/baseline_mixture.RULE_TEXT")
    C = load_json("results", "frontier_panel", "closed_w.json")
    J = load_json("results", "ladder", "analysis", "ladder.json")
    W, notes = {}, []
    for t in TASKS:
        x = B["tasks"][t]
        W[t] = {k: float(w) for k, w in zip(x["models"], x["w_discovery"])}
        lp = {r["model_key"]: r["w"] for r in J["panel"][t]}
        if set(lp) != set(W[t]) or any(abs(lp[k] - w) > TOL for k, w in W[t].items()):
            raise SystemExit(f"{t}: panel w differs between baseline_mixture.json and ladder.json's panel block")
        notes.append(f"{T_NAME[t]}: panel w of baseline_mixture.json equal ladder.json's panel block, "
                     f"{len(W[t])} models (1e-9)")
        for k in LADDER6:
            tk = J["models"][k]["tasks"][t]
            wc, wu = float(tk["cut"]["w"]["discovery"]), float(tk["uncut"]["w"]["discovery"])
            if abs(wc - wu) > TOL:
                raise SystemExit(f"{t} {k}: w differs between the cut and uncut readings")
            W[t][k] = wc
        rows = [r for r in C["per_benchmark"] if r.get("model") == CLOSED and r.get("task") == t
                and r.get("arm") == "requested"]
        if len(rows) != 1:
            raise SystemExit(f"{t}: {len(rows)} Qwen4-Max requested rows in closed_w.json")
        W[t][CLOSED] = float(rows[0]["w16_api_discovery"])
    notes.append("ladder w (ladder.json, discovery draws, the panel's items and templates) equal in the cut and uncut "
                 "readings (1e-9); Qwen4-Max's w is on its 100 items per benchmark (closed_w.json's panel block holds "
                 "the panel's w on those 100 items, which this analysis does not use)")
    return W, notes


def w_block(W: dict, D: dict, keys: list[str], n_boot: int | None, label: str) -> dict:
    """rho of w with date (with its family bootstrap if n_boot), and the recent and older means, over keys."""
    from src import sampling
    E = D["entries"]
    x = [ordinal(E[k]["date"]) for k in keys]
    y = [W[k] for k in keys]
    fams = [E[k]["family"] for k in keys]
    out = {"M": len(keys), "rho": spearman(x, y), "families": sorted(set(fams))}
    if n_boot:
        out["family_bootstrap"] = family_bootstrap_rho(x, y, fams, n_boot, sampling.rng_for("release_dates", label))
    rec = [k for k in keys if E[k]["date"] >= RECENT_FROM]
    old = [k for k in keys if E[k]["date"] < RECENT_FROM]
    out["recent"] = {"M": len(rec), "mean_w": float(np.mean([W[k] for k in rec])) if rec else None, "models": rec}
    out["older"] = {"M": len(old), "mean_w": float(np.mean([W[k] for k in old])) if old else None, "models": old}
    out["date_ties"] = sum(1 for i in range(len(x)) for j in range(i + 1, len(x)) if x[i] == x[j])
    return out


def pair_counts(pairs: list[dict], W: dict) -> dict:
    rows, hi, lo, tie = [], 0, 0, 0
    for p in pairs:
        o, n = p["older"], p["newer"]
        if o not in W or n not in W:
            continue
        d = W[n] - W[o]
        r = "higher" if d > TOL else "lower" if d < -TOL else "tie"
        hi, lo, tie = hi + (r == "higher"), lo + (r == "lower"), tie + (r == "tie")
        rows.append({"older": o, "newer": n, "w_older": W[o], "w_newer": W[n], "newer_minus_older": d, "result": r})
    return {"pairs": len(rows), "newer_higher": hi, "newer_lower": lo, "ties": tie, "rows": rows}


def recent_reversals(D: dict) -> dict:
    """The reversed share among the checkpoints from 2025 on, with model and family bootstraps."""
    import robustness_inputs as ri
    import run_decision_directions as rdd
    from src import decision_directions as dd, robustness as rb, sampling
    if rdd.N_BOOT != N_BOOT:
        raise SystemExit("scripts/run_decision_directions.N_BOOT differs from this script's")
    WB = load_json("results", "statement_values", "statement_values.json")["recent"]
    AE = load_json("results", "arm_effect", "arm_effect.json")["tasks"]
    E = D["entries"]
    out = {}
    for t in TASKS:
        x = ri.load(t)
        f, d = x["facts"], x["discovery"]
        keys, fams = list(f["panel"]), list(f["families"])
        req, fd, cot = (d[a].mean(0) for a in ("requested", "forced", "cot"))
        g1, g2 = cot - req, cot - fd
        full = rb.pair_reversals(g1, g2)
        if (full["reversed"], full["pairs"]) != (AE[t]["reversals"]["reversed"], AE[t]["reversals"]["pairs"]):
            raise SystemExit(f"{t}: the full panel's count differs from results/arm_effect/arm_effect.json")
        bad = [k for k, fm in zip(keys, fams) if E[k]["family"] != fm]
        if bad:
            raise SystemExit(f"{t}: families differ from the dates manifest for {bad}")
        sel = np.array([j for j, k in enumerate(keys) if E[k]["date"] >= RECENT_FROM])
        rk = [keys[j] for j in sel]
        st = WB["tasks"][t]
        if rk != st["models"]:
            raise SystemExit(f"{t}: the recent models differ from statement_values.json's")
        u = dd.reversal_uncertainty(g1[sel], g2[sel], [fams[j] for j in sel], N_BOOT,
                                    sampling.rng_for("release_dates", "recent_reversal", t))
        if (u["reversed"], u["pairs"]) != (st["reversed"], st["pairs"]) or abs(u["share"] - st["share"]) > 1e-12:
            raise SystemExit(f"{t}: {u['reversed']} of {u['pairs']} differs from the stored {st['reversed']} of "
                             f"{st['pairs']}")
        tg = T_TAG[t]
        stored = load_json("results", "statement_values", "statement_values.json")["macros"][f"aeRecent{tg}ReversedPct"]
        if num(100 * u["share"], 0) != stored:
            raise SystemExit(f"{t}: the printed share differs from \\aeRecent{tg}ReversedPct ({stored})")
        fam_count = {fm: sum(1 for j in sel if fams[j] == fm) for fm in sorted({fams[j] for j in sel})}
        out[t] = {"M": len(rk), "models": rk, "families": fam_count, "reversed": u["reversed"], "pairs": u["pairs"],
                  "share": u["share"], "ties": rb.pair_reversals(g1[sel], g2[sel])["ties"],
                  "model_bootstrap": u["model_bootstrap"], "family_bootstrap": u["family_bootstrap"],
                  "leave_one_family_out": u["leave_one_family_out"], "within_family": u["within_family"],
                  "between_family": u["between_family"], "stored_pct": stored}
    return out


def run() -> dict:
    if not (os.path.exists(DATES) and os.path.exists(PAIRS)):
        raise SystemExit("run `dates` and `pairs` first: the manifests are written before w is read")
    D = load_json("manifests", "panel_release_dates.json")
    PR = load_json("manifests", "panel_release_pairs.json")
    if PR["dates_written_at"] != D["written_at"]:
        raise SystemExit("the pairs manifest was written from another dates manifest")
    W, notes = load_w()
    pk = panel_keys()
    R = {"request": REQUEST, "script": "scripts/release_dates.py run", "created": time.strftime("%Y-%m-%d %H:%M"),
         "dates_manifest": {"path": rel(DATES), "written_at": D["written_at"]},
         "pairs_manifest": {"path": rel(PAIRS), "written_at": PR["written_at"]},
         "w_definition": "the share of a model's requested-direct replies longer than 16 generated tokens, per "
                         "configuration, averaged over the 24 discovery draws (src/baseline_mixture.RULE_TEXT)",
         "checks": notes, "n_boot": N_BOOT, "recent_from": RECENT_FROM, "tasks": {}, "extension": {}}
    m = {}
    for t in TASKS:
        keys = pk[t]
        b = w_block(W[t], D, keys, N_BOOT, f"rho/{t}")
        b["pairs"] = pair_counts(PR["pairs"], {k: W[t][k] for k in keys})
        b["pairs_alt"] = pair_counts(PR["pairs_alt"], {k: W[t][k] for k in keys})
        # sensitivity: the panel checkpoint with the latest date before the cut counted as recent (the choice of scripts/statement_values.py)
        E = D["entries"]
        near = max((k for k in keys if E[k]["date"] < RECENT_FROM), key=lambda k: (E[k]["date"], k))
        rec2 = [k for k in keys if E[k]["date"] >= RECENT_FROM] + [near]
        old2 = [k for k in keys if k not in rec2]
        b["latest_before_cut_moved"] = {"model": near, "date": E[near]["date"],
                                        "recent_mean_w": float(np.mean([W[t][k] for k in rec2])),
                                        "older_mean_w": float(np.mean([W[t][k] for k in old2]))}
        b["per_model"] = {k: {"date": E[k]["date"], "family": E[k]["family"], "w": W[t][k]} for k in keys}
        R["tasks"][t] = b
        ext_keys = keys + [k for k in list(LADDER6) + [CLOSED] if k in W[t]]
        xb = w_block(W[t], D, ext_keys, None, f"ext/{t}")
        xb["pairs"] = pair_counts(PR["pairs"] + PR["extension_pairs"], {k: W[t][k] for k in ext_keys})
        xb["added"] = {k: {"date": D["entries"][k]["date"], "basis": D["entries"][k]["basis"], "w": W[t][k]}
                       for k in ext_keys if k not in keys}
        R["extension"][t] = xb
        tg = T_TAG[t]
        fb = b["family_bootstrap"]["ci95"] or [None, None]
        m[f"wDateRho{tg}"] = num(b["rho"], 2)
        m[f"wDateRho{tg}Lo"] = num(fb[0], 2)
        m[f"wDateRho{tg}Hi"] = num(fb[1], 2)
        m[f"wRecentMean{tg}Pct"] = num(100 * b["recent"]["mean_w"], 1)
        m[f"wOlderMean{tg}Pct"] = num(100 * b["older"]["mean_w"], 1)
    for t in ("gsm8k", "math"):
        m[f"wNewerHigher{T_TAG[t]}"] = str(R["tasks"][t]["pairs"]["newer_higher"])
    if R["tasks"]["gsm8k"]["pairs"]["pairs"] != R["tasks"]["math"]["pairs"]["pairs"]:
        raise SystemExit("GSM8K and MATH-500 count different pairs")
    m["wNewerPairs"] = str(R["tasks"]["gsm8k"]["pairs"]["pairs"])
    RV = recent_reversals(D)
    R["recent_reversals"] = RV
    for t, x in RV.items():
        tg = T_TAG[t]
        mb = x["model_bootstrap"]["ci95"] or [None, None]
        fb = x["family_bootstrap"]["ci95"] or [None, None]
        m[f"aeRecent{tg}ReversedPctLo"] = num(None if mb[0] is None else 100 * mb[0], 0)
        m[f"aeRecent{tg}ReversedPctHi"] = num(None if mb[1] is None else 100 * mb[1], 0)
        m[f"aeRecent{tg}ReversedPctFamLo"] = num(None if fb[0] is None else 100 * fb[0], 0)
        m[f"aeRecent{tg}ReversedPctFamHi"] = num(None if fb[1] is None else 100 * fb[1], 0)
    R["macros"] = m
    R["dates"] = {k: {kk: e[kk] for kk in ("group", "family", "date", "basis")} for k, e in D["entries"].items()}
    R["flags"] = D["flags"]
    return R


# ============================================================================ summary


def f2(x, nd=2):
    return "undefined" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def summary(R: dict) -> str:
    L = ["# Does w rise with release date? Intervals for the newest-model reversal shares",
         "",
         f"Analysis: {REQUEST}; `scripts/release_dates.py`, {R['created']}.",
         "",
         f"w: {R['w_definition']}. Release dates: `{R['dates_manifest']['path']}` (written "
         f"{R['dates_manifest']['written_at']}); pairs: `{R['pairs_manifest']['path']}` (written "
         f"{R['pairs_manifest']['written_at']}); both written before this script read any w. Recent: released on or "
         f"after {R['recent_from']} by the Hugging Face repository creation date, the 14 checkpoints of "
         "\\aeRecentM (13 on BIG-Bench Hard, where gemma-4-31b-it has no rows).",
         "",
         "Checks: " + "; ".join(R["checks"]) + ".",
         "",
         "## w against release date, panel",
         "",
         f"Spearman rho of w with the release date (days; tied dates take average ranks). Interval: 95% percentile, "
         f"{R['n_boot']:,} draws of whole families with replacement (five families). Means: plain means of the models' w.",
         "",
         "| Benchmark | M | rho | 95% family interval | undefined draws | rho leaving one family out | Mean w, 2025 on (M) | Mean w, earlier (M) |",
         "|---|---:|---:|---|---:|---|---|---|"]
    for t, b in R["tasks"].items():
        fb = b["family_bootstrap"]
        ci = fb["ci95"]
        lofo = ", ".join(f"{f} {f2(v)}" for f, v in fb["leave_one_family_out"].items())
        L.append(f"| {T_NAME[t]} | {b['M']} | {f2(b['rho'])} | "
                 f"{'[' + f2(ci[0]) + ', ' + f2(ci[1]) + ']' if ci else 'undefined'} | {fb['n_undefined']} | {lofo} | "
                 f"{f2(100 * b['recent']['mean_w'], 1)}% ({b['recent']['M']}) | "
                 f"{f2(100 * b['older']['mean_w'], 1)}% ({b['older']['M']}) |")
    L += ["", "Sensitivity of the means: the panel checkpoint with the latest date before the cut counted as recent "
          "(the choice of `scripts/statement_values.py`):"]
    for t, b in R["tasks"].items():
        s = b["latest_before_cut_moved"]
        L.append(f"- {T_NAME[t]}: {s['model']} ({s['date']}) moved: 2025 on {f2(100 * s['recent_mean_w'], 1)}%, earlier "
                 f"{f2(100 * s['older_mean_w'], 1)}%.")
    L += ["", "## Within families, newer against older release", "",
          "Rule (`manifests/panel_release_pairs.json`, written before w was read): " + R.get("pair_rule", ""), ""]
    L += ["| Benchmark | Pairs | Newer higher | Newer lower | Ties | Wider band (x2): pairs, newer higher, lower, ties |",
          "|---|---:|---:|---:|---:|---|"]
    for t, b in R["tasks"].items():
        p, a = b["pairs"], b["pairs_alt"]
        L.append(f"| {T_NAME[t]} | {p['pairs']} | {p['newer_higher']} | {p['newer_lower']} | {p['ties']} | "
                 f"{a['pairs']}, {a['newer_higher']}, {a['newer_lower']}, {a['ties']} |")
    L += ["", "Pairs, w in % (older → newer):", ""]
    L += ["| Older | Newer | GSM8K | MATH-500 | BIG-Bench Hard |", "|---|---|---|---|---|"]
    rows = {}
    for t, b in R["tasks"].items():
        for r in b["pairs"]["rows"]:
            rows.setdefault((r["older"], r["newer"]), {})[t] = r
    for (o, n), rr in rows.items():
        cells = []
        for t in TASKS:
            r = rr.get(t)
            cells.append(f"{f2(100 * r['w_older'], 1)} → {f2(100 * r['w_newer'], 1)} ({r['result']})" if r else "absent")
        L.append(f"| {o} ({R['dates'][o]['date']}) | {n} ({R['dates'][n]['date']}) | " + " | ".join(cells) + " |")
    near = [(t, r) for t, b in R["tasks"].items() for r in b["pairs"]["rows"] if abs(r["newer_minus_older"]) < 0.005]
    L += ["", "Pairs decided by less than 0.5 points of w: " + ("; ".join(
        f"{T_NAME[t]}, {r['older']} → {r['newer']}, newer − older {100 * r['newer_minus_older']:+.3f} points ({r['result']})"
        for t, r in near) if near else "none") + "."]
    L += ["", "## The extension (descriptive, no intervals)", "",
          "Added: the six size-ladder models (w from `results/ladder/analysis/ladder.json`, the panel's items and "
          "templates) and Qwen4-Max (w from `results/frontier_panel/closed_w.json`, API count, 100 items per "
          "benchmark, discovery draws). Dates and their bases:", ""]
    for k, x in R["extension"]["gsm8k"]["added"].items():
        L.append(f"- {k}: {x['date']} ({x['basis']})")
    L += ["", "| Benchmark | M | rho | Mean w, 2025 on (M) | Mean w, earlier (M) | Pairs | Newer higher | Newer lower | Ties |",
          "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for t, b in R["extension"].items():
        p = b["pairs"]
        L.append(f"| {T_NAME[t]} | {b['M']} | {f2(b['rho'])} | {f2(100 * b['recent']['mean_w'], 1)}% "
                 f"({b['recent']['M']}) | {f2(100 * b['older']['mean_w'], 1)}% ({b['older']['M']}) | {p['pairs']} | "
                 f"{p['newer_higher']} | {p['newer_lower']} | {p['ties']} |")
    ext_rows = [r for r in R["extension"]["gsm8k"]["pairs"]["rows"]
                if r["newer"] in R["extension"]["gsm8k"]["added"] or r["older"] in R["extension"]["gsm8k"]["added"]]
    if ext_rows:
        L += ["", "Pairs the extension adds (w in %, GSM8K, MATH-500, BIG-Bench Hard):"]
        for r in ext_rows:
            cells = []
            for t in TASKS:
                rr = next((x for x in R["extension"][t]["pairs"]["rows"]
                           if x["older"] == r["older"] and x["newer"] == r["newer"]), None)
                cells.append(f"{f2(100 * rr['w_older'], 1)} → {f2(100 * rr['w_newer'], 1)}" if rr else "absent")
            L.append(f"- {r['older']} → {r['newer']}: " + "; ".join(cells) + ".")
    L += ["", "## Intervals for the newest-model reversal shares", "",
          f"Reversed pairs (CoT − requested against CoT − forced, strict, model means over the 24 discovery draws) among "
          f"the checkpoints from 2025 on; `src/decision_directions.reversal_uncertainty`, {R['n_boot']:,} draws: models "
          "drawn with replacement (a pair of a model with its own copy left out), and families drawn whole. 95% "
          "percentile intervals; draws with no pair are undefined and left out (counted).", "",
          "| Benchmark | M | Families (models) | Reversed / pairs | Share | Model interval | Family interval | Undefined (model, family) |",
          "|---|---:|---|---|---:|---|---|---|"]
    for t, x in R["recent_reversals"].items():
        mb, fb = x["model_bootstrap"], x["family_bootstrap"]
        fam = ", ".join(f"{f} {n}" for f, n in x["families"].items())
        ci = lambda c: f"[{f2(100 * c[0], 1)}, {f2(100 * c[1], 1)}]" if c else "undefined"  # noqa: E731
        L.append(f"| {T_NAME[t]} | {x['M']} | {fam} | {x['reversed']} / {x['pairs']} | {f2(100 * x['share'], 1)}% "
                 f"(macro {x['stored_pct']}) | {ci(mb['ci95'])} | {ci(fb['ci95'])} | {mb['n_undefined']}, "
                 f"{fb['n_undefined']} |")
    L += ["", "Only three families have checkpoints from 2025 on, so the family bootstrap draws three families each time "
          "and its interval is coarse.", "", "## Dates: notes", ""]
    for fl in R["flags"]:
        L.append(f"- {fl['model']}: {fl['what']} ({fl['from']}).")
    L += ["", "## Macros", "", "| Name | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(R["macros"].items())]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("dates", "pairs", "run"))
    ap.add_argument("--force", action="store_true", help="overwrite a write-once manifest")
    a = ap.parse_args(argv)
    if a.stage == "dates":
        D = build_dates()
        print(rel(DATES), write_once(DATES, D, ("entries", "flags", "panel_rule", "extension_rule"), a.force))
        return 0
    if a.stage == "pairs":
        if not os.path.exists(DATES):
            raise SystemExit("run `dates` first")
        P = build_pairs(load_json("manifests", "panel_release_dates.json"))
        print(rel(PAIRS), write_once(PAIRS, P, ("rule", "rule_alt", "pairs", "pairs_alt", "extension_pairs"), a.force))
        for p in P["pairs"]:
            print(f"  {p['older']} ({p['older_date']}) -> {p['newer']} ({p['newer_date']})")
        print(f"  {len(P['pairs'])} pairs; wider band {len(P['pairs_alt'])}; extension adds {len(P['extension_pairs'])}")
        return 0
    R = run()
    R["pair_rule"] = load_json("manifests", "panel_release_pairs.json")["rule"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{STEM}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(R), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    txt = summary(R)
    for name in (f"{STEM}.md", f"{STEM}_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
