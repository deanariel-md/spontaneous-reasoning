r"""Independent check of the size ladder's analysis (results/ladder/analysis/ladder.json, ladder.md and ladder_note.md,
written by scripts/ladder_analysis.py).

Written apart from the analysis: it imports neither scripts/ladder_analysis.py nor the helpers that script uses
(scripts/nebius_panel_report.py, scripts/nebius_panel.py, src/). It shares with the analysis the labels the text uses
(model, benchmark and family names) and the design (which models form which family ladder). Its rules are its own
constants (direction, marks, ties, rounding; ladder.json `rules` states the same in words and is never read as a rule)
or are read from the source that states them, as text, never imported: the corrected gain from
src/baseline_mixture.RULE_TEXT (verify_ladder_ground_truth.corrected_gain_rule: the cap comparison, the draws of w and
of the gain, the formula; a part the text does not state is a mismatch), and the forced replies' completion rule from
src/forced_stop.py (its lists of wrappers and continuation words and signs, and its version).

Part 3, ground truth (scripts/verify_ladder_ground_truth.py, its own code). Parts 1 and 2 recompute
from the scored rows, so on their own they cannot see an error in the scoring. Part 3
goes back to the raw replies: (a) the forced replies the stop at the first "}" left open, completed or unread, per
ladder model, family rung, benchmark and arm, recounted with the rule as stated, against ladder.json (`stop_counts`,
`stop_unread`), and the family steps' bound (rules.direction, "within the stop's loss") from these counts; (b) the two
API models' scored forced replies that the cut leaves open, against the box the model closed in its whole reply, read
here (a model with more than 1% disagreeing is a mismatch); (c) a seeded sample of raw replies per ladder model,
benchmark (GSM8K, MATH-500, BIG-Bench Hard, HotpotQA), reading (uncut: the text as generated; cut: the copy cut at the
panel's cap) and arm: 150 replies, and in the cut reading up to 50 more replies that the cap cut, drawn apart as a
second stratum (verify_ladder_ground_truth.SAMPLE_N, SAMPLE_CUT_N, seed SAMPLE_SEED), each read here and compared with
the scored rows' content_correct of that reading. A cell fails at its first disagreement (SAMPLE_MAX_DISAGREE = 0)
and every disagreement is printed; each cell also prints a one-sided 95% upper bound on
its share of disagreements. Tests plant a reading error in the scored rows and in the reader and see this part fail.
Without raw replies under the root (the tests' synthetic roots) part 3 does not run, and the VERIFY line says so.

Part 4, orderings (scripts/ladder_orderings.py), when results/ladder/analysis/
ladder_orderings.json exists. From the six ladder models' exact means recomputed in part 1 (discovery draws; HotpotQA
content too, the paper's basis of its count) and the panel's result files read here (panel_tables; HotpotQA content's
discovery-draw means from results/hotpot_content/hotpot_content.json), with its own pair loop (a pair tied under either
gain, |difference| <= 1e-9 pp, is neither reversed nor kept and stays among the pairs): the reversed pairs among the six
(named), the four of 70B and above, the combined panel of 35 (34 on BIG-Bench Hard) and the 29 panel models, per
benchmark and pooled over the first three, in both readings and both scores, and on the 48 draws for the hand
counts (HAND_COUNTS); the model bootstrap intervals by its own resampling with its own seed (each bound
within 0.01 of the file's); w per model and the family directions (its own rule); the baseline shift split at 16
tokens, per configuration from the rows (split_parts, inside recompute); the panel's ranges from the files the
orderings cite; Qwen3.5-122B-A10B's unreadable requested replies read from its raw replies (no box: its own test for
\boxed; forms by its own patterns). Every lad... and comb... macro of paper/numbers.tex is compared, as text, with its
own value printed half-even, and every line of ladder_note.md's section "Orderings among the ladder models" is read
(TextCheck.orderings). ladder_orderings.md itself is not parsed: it is rendered from the JSON, which is checked.

Part 1, numbers in ladder.json. From the scored parquet rows alone, with the sampler manifests and the ladder's decoding
files, it recomputes per ladder model, benchmark, reading (cut, uncut) and score (strict, content):
  - the zero-shot arm means (requested, forced, CoT) and the gains CoT - requested and CoT - forced, on the
    (configuration, item) units that have all three zero-shot arms, as the mean over a draw list of the
    per-configuration accuracy: the sampler's 48 draws (discovery then validation; a configuration drawn twice counts
    twice) and the 24 discovery draws; and pooled over every row of those units;
  - w, the share of requested outputs longer than the forced cap, per configuration and averaged over the draws;
  - the corrected gain, field by field as ladder.json's rules state it: discovery = (1 - w over the discovery draws) x
    the discovery-draw CoT - forced (the rule's, src/baseline_mixture.RULE_TEXT); pooled = (1 - discovery w) x the
    pooled CoT - forced; all_draws = (1 - w over the 48 draws) x the 48-draw CoT - forced (the definition of
    results/frontier_panel/nebius_panel.json; printed nowhere);
  - readable replies per arm (the strict scorer's `compliant` flag: per-configuration share over the discovery and the
    48 draws, and over all rows), content - strict per arm over all rows and on the discovery draws;
  - rows per arm, outputs cut at the panel's cap per arm, and the five-shot arms;
then the family tables (values and directions), the placements against the panel (both readings), the BIG-Bench Hard
forced gaps, the cut-against-uncut summary, completeness against the request files, and the DeepSeek reproduction;
and DeepSeek's printed tables in results/frontier_panel/nebius_panel.md. Each is compared with ladder.json (tolerance
1e-9) or with the printed digits. The cut-against-uncut summary runs over the quantities ladder.md prints, read from
its own tables (printed_lists: section 6's rows and the glance tables' placed columns, the residual and the readable
shares included; section 5's rows for the family directions), never over a list repeated from the analysis.

Part 2, what is printed. It reads every line and every table of ladder.md (the lines above section 1 and sections 1
to 13) and the whole of ladder_note.md. Each line is masked for labels and references (model names, file names,
item letters), its numbers are checked by the handler of its section, and any number left over is a mismatch ("not
checked"); a table no handler reads is a mismatch. It compares every number at its printed precision with its own
values, every direction word with its own reading of the rule (`rules.direction`), every up or down mark and every
tie count ("k tied") with its own placement, every footnote number with the source check the footnote cites, and the
sizes, spend, panel precision and check counts with the records the text cites, read here on their own. It recomputes
the template SDs of section 7 from the rows (exact variance over configurations, minus the configuration x item
interaction mean square over the items, by the two-way layout's sums of squares) and checks each printed SD by
bounds on its exact square. Pairs of ladder models are a reversal only when both readings
order them, in opposite ways; a pair tied in one reading is counted apart. The scorer part of the five-shot basis
check is recomputed here: Mistral Large 3's saved rows of the arm, counted here, against the file's own scoring
of them, never the file's flag alone.

Rounding (ladder.json `rules.rounding`), checked independently: every value this script recomputes is an exact ratio
(fractions.Fraction from integer counts: sums of 0/1 columns over rows, per configuration, over draws); each panel
value is the fraction with the smallest denominator within the design's bound that its file's float stands for
(Fraction.limit_denominator, with the bound draws x items per configuration, or the arm's rows for pooled values, and
a check that it is within 1e-9 of the float); the BIG-Bench Hard five-shot panel gaps are the decimals the file gives.
A printed number must equal, as a string, Python's own half-even formatting of that exact value
(format(Fraction, '.<d>f')). A value exactly halfway therefore passes only as its even neighbour (8.95 as 9.0, never
8.9). A value held only as a float within 1e-9 of a halfway point is a mismatch: its digit cannot be settled.

Prints every mismatch and a last line `VERIFY: <n> values checked, <m> mismatches; ground truth: ...` (the sample's
and the full replies' agreement). Exit code 0 when m is 0.

Run from the repository root:
    PYTHONIOENCODING=utf-8 python scripts/verify_ladder_analysis.py
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# the ground-truth stage, written for this checker; it reads the raw replies with its own code
import verify_ladder_ground_truth as GT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["gsm8k", "math", "bbh", "hotpot"]
DISC_TASKS = ["gsm8k", "math", "bbh"]
READINGS = ["cut", "uncut"]
TOL = 1e-9
SCORE_COL = {"strict": "strict_correct", "content": "content_correct"}
ZERO_ARMS = {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot"}
LABEL = {"direct_answer": "requested", "forced_direct": "forced", "zero_shot_cot": "cot",
         "fewshot_forced": "fs_forced", "fewshot_requested": "fs_requested"}
TASK_ROW = {"GSM8K": "gsm8k", "MATH-500": "math", "BIG-Bench Hard": "bbh", "HotpotQA": "hotpot"}
TASK_NAME = {v: k for k, v in TASK_ROW.items()}

# Where each ladder model's scored rows are (the layouts of scripts/score_ladder.py, scripts/mistral_ladder.py and
# scripts/nebius_panel_report.py).
CLUSTER = {"cut": "data/processed/ladder/{task}/{key}.parquet",
           "uncut": "data/processed/ladder_uncut/{task}/{key}.parquet"}
FILES = {
    "ministral-3-8b-instruct-2512": CLUSTER,
    "ministral-3-14b-instruct-2512": CLUSTER,
    "mistral-large-2512": {"cut": "data/ladder_api/mistral-large-2512/scored/{task}_cut.parquet",
                           "uncut": "data/ladder_api/mistral-large-2512/scored/{task}_uncut.parquet"},
    "llama-3.1-70b-instruct-fp8": CLUSTER,
    "qwen3.5-122b-a10b-fp8": CLUSTER,
    "deepseek-v4.1-flash-1017": {"cut": "data/frontier_panel/scored/{task}_scored.parquet",
                                 "uncut": "data/frontier_panel/scored/{task}_scored_uncut.parquet"},
}
API = {"mistral-large-2512", "deepseek-v4.1-flash-1017"}
API_DESIGN_KEY = "ministral-3-14b-instruct-2512"
DS = "deepseek-v4.1-flash-1017"
# Labels the text uses: (name in tables, name in sentences)
NAMES = {"ministral-3-8b-instruct-2512": ("Ministral 3 8B Instruct 2512", "Ministral 3 8B Instruct 2512"),
         "ministral-3-14b-instruct-2512": ("Ministral 3 14B Instruct 2512", "Ministral 3 14B Instruct 2512"),
         "mistral-large-2512": ("Mistral Large 3 (2512)", "Mistral Large 3"),
         "llama-3.1-70b-instruct-fp8": ("Llama-3.1-70B-Instruct", "Llama-3.1-70B-Instruct"),
         "qwen3.5-122b-a10b-fp8": ("Qwen3.5-122B-A10B", "Qwen3.5-122B-A10B"),
         "deepseek-v4.1-flash-1017": ("DeepSeek V4.1-Flash-1017", "DeepSeek V4.1-Flash-1017")}
KEY_OF = {n: k for k, ns in NAMES.items() for n in ns}
# The family ladders of the design: rungs smallest first, panel or ladder.
FAMILIES = {
    "Mistral 3": {"rungs": [("ministral-3-8b-instruct-2512", "ladder"), ("ministral-3-14b-instruct-2512", "ladder"),
                            ("mistral-large-2512", "ladder")],
                  "context": ["mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3", "mistral-nemo-instruct-2407",
                              "mistral-small-3.1-24b-instruct-2503"]},
    "Qwen3.5": {"rungs": [("qwen3.5-9b", "panel"), ("qwen3.5-35b-a3b", "panel"), ("qwen3.5-122b-a10b-fp8", "ladder")],
                "context": []},
    "Llama 3.1": {"rungs": [("llama-3.1-8b-instruct", "panel"), ("llama-3.1-70b-instruct-fp8", "ladder")],
                  "context": []},
}
FAMILY_OF = {k: f for f, s in FAMILIES.items() for k, src in s["rungs"] if src == "ladder"}
SIZE = {"ministral-3-8b-instruct-2512": "8B", "ministral-3-14b-instruct-2512": "14B",
        "mistral-large-2512": "Large 3 (675B)", "qwen3.5-9b": "9B", "qwen3.5-35b-a3b": "35B-A3B",
        "qwen3.5-122b-a10b-fp8": "122B-A10B", "llama-3.1-8b-instruct": "8B", "llama-3.1-70b-instruct-fp8": "70B"}
PANEL_RUNGS = {k for s in FAMILIES.values() for k, src in s["rungs"] if src == "panel"} | \
    {k for s in FAMILIES.values() for k in s["context"]}
QUANTS = ["requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced", "w", "corrected_gain", "residual"]
POOLED_CG = "(1 − discovery w) × pooled CoT − forced"
Q_OF = {"Requested": "requested", "Forced": "forced", "CoT": "cot", "CoT − requested": "cot_minus_requested",
        "CoT − forced": "cot_minus_forced", "w": "w", "Corrected gain": "corrected_gain",
        "CoT − requested minus corrected gain": "residual", POOLED_CG: "corrected_gain",
        f"CoT − requested minus {POOLED_CG}": "residual"}
BASIS_WORDS = {"discovery": "discovery draws", "pooled": "pooled rows"}
# The glance tables' placed columns (section 1): (score, quantity).
GLANCE_COLS = {"CoT − forced, strict": ("strict", "cot_minus_forced"), "CoT − forced, content": ("content",
                                                                                              "cot_minus_forced"),
               "CoT − requested, strict": ("strict", "cot_minus_requested"),
               "CoT − requested, content": ("content", "cot_minus_requested"), "w": ("strict", "w"),
               "Residual, strict": ("strict", "residual"), "Residual, content": ("content", "residual")}
READ_ARMS = ["requested", "forced", "cot"]
# How the text names the placed quantities the cut-against-uncut comparison covers (it must say what is compared).
PLACED_WORDS = ("section 6's rows: per score the three arms, the two gains, w, the corrected gain and the residual; "
                "the readable share of each zero-shot arm on GSM8K, MATH-500 and BIG-Bench Hard")
PILOT_MIN = 90       # scripts/check_ladder_pilot.py: a boxed answer in at least 90% of the requested arms' rows
PILOT_MD = "results/ladder/pilot_check_qwen3.5-122b-a10b-fp8.md"
# The rules the checker applies itself (stated in ladder.json, `rules`; the text quotes them).
FLAT = {"pp": Fraction(1, 10), "w": Fraction(1, 1000)}
RESIDUAL_BAND, READABLE_BELOW, CUT_BIG, FORM_MIN = Fraction(2), Fraction(90), Fraction(1), Fraction(1, 20)
# The panel's design: items per configuration (every configuration applied to every target item; results/ladder/
# plan.md and the request files); with the sampler's draws this bounds the denominator of each panel value.
PANEL_ITEMS = {"gsm8k": 1319, "math": 398, "bbh": 1350, "hotpot": 1000}
# The complete design's rows per benchmark and arm (results/ladder/plan.md; src/ladder.py), against which an edge case
# "rows per arm differ from the design" is stated.
DESIGN = {"gsm8k": {"direct_answer": 38251, "zero_shot_cot": 38251, "forced_direct": 38251, "fewshot_forced": 38251,
                    "fewshot_requested": 10552},
          "math": {"direct_answer": 12338, "zero_shot_cot": 12338, "forced_direct": 12338, "fewshot_forced": 12338},
          "bbh": {"direct_answer": 47250, "zero_shot_cot": 47250, "forced_direct": 47250, "fewshot_forced": 47250,
                  "fewshot_requested": 10800},
          "hotpot": {a: 36000 for a in ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced")}}
FEWSHOT_PANEL = "results/ladder/mistral_bbh_forced_check.json"     # panel_scan: the panel's five-shot forced arm
FEWSHOT_ROWS = 47250                                                 # BIG-Bench Hard rows of an arm (35 x 1,350)
LOOP_LOG = "results/ladder/auto_ladder2.log"
FORMS_JSON = "results/ladder/analysis/bbh_forced_forms.json"
MISTRAL_CHECK_JSON = "results/ladder/mistral_bbh_forced_check.json"         # F1, Mistral Large 3
NEBIUS_CHECK_JSON = "results/frontier_panel/nebius_forced_check.json"  # F1, DeepSeek
NEBIUS_CHECK = "results/frontier_panel/nebius_forced_check_clean.md"
PILOT_QWEN = "results/ladder/pilot_check_qwen3.5-122b-a10b-fp8.json"
NEBIUS_JSON = "results/frontier_panel/nebius_panel.json"
STOP_AUDIT_JSON = "results/forced_stop/audit.json"          # F1, the stop's loss (scripts/forced_stop_audit.py)
FULL_REPLIES_JSON = "results/forced_stop/full_replies.json"  # F1, the API models' full replies
# Orderings on the size ladder (scripts/ladder_orderings.py writes ladder_orderings.json,
# the macros lad... and comb... of paper/numbers.tex and a section of ladder_note.md), recounted here (check_orderings).
ORD_JSON = "ladder_orderings.json"
ORD_SECTION = "Orderings among the ladder models"
HAND_COUNTS = {"gsm8k": (2, 15), "math": (4, 15), "bbh": (2, 15), "hotpot": (12, 15)}   # counted by hand (content, 48-draw means)
NUMBERS_TEX = "paper/numbers.tex"
SPLIT_TOKENS = 16                 # the shift split at 16 generated tokens (the forced cap of the first three)
BIG = ["llama-3.1-70b-instruct-fp8", "qwen3.5-122b-a10b-fp8", "mistral-large-2512", "deepseek-v4.1-flash-1017"]
QWEN = "qwen3.5-122b-a10b-fp8"
ORD_BOOT, ORD_SEED, ORD_BOOT_TOL = 9999, 20261124, 0.01   # own resampling, own seed; bounds within 0.01 of the file's
ORD_GAPS = (1, 2, 5, 10)        # pp: pairs whose two gaps both exceed each, and the reversed among them
ORD_GAP_WORD = {1: "One", 2: "Two", 5: "Five", 10: "Ten"}
ORD_SUB_PP = 1                  # the reversed pairs of the six that the note names: a gap this small under either
ORD_T = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh", "hotpot": "Hq", "pooled": "Pooled"}
ORD_MODEL = {"ministral-3-8b-instruct-2512": "MinistralEight", "ministral-3-14b-instruct-2512": "MinistralFourteen",
             "mistral-large-2512": "MistralLarge", "llama-3.1-70b-instruct-fp8": "LlamaSeventy",
             "qwen3.5-122b-a10b-fp8": "QwenOneTwentyTwo", "deepseek-v4.1-flash-1017": "DeepSeek",
             "qwen3.5-9b": "QwenNine", "qwen3.5-35b-a3b": "QwenThirtyFive", "llama-3.1-8b-instruct": "LlamaEight"}
ORD_FAM = {"Mistral 3": "Mistral", "Qwen3.5": "Qwen", "Llama 3.1": "Llama"}
ORD_VARIANT = {("content", "cut"): "", ("strict", "cut"): "Strict", ("content", "uncut"): "Uncut",
               ("strict", "uncut"): "StrictUncut"}


def gap_json(root: Path, rel: str) -> dict | None:
    """A forced-arm check's JSON when it holds the gap by form that F1 quotes (read here, apart from the analysis)."""
    p = root / rel
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    return g if isinstance(g.get("gap"), dict) and "by_form" in g["gap"] else None


# ----------------------------------------------------------------------------------------------------------------
# Recomputation from scored rows
# ----------------------------------------------------------------------------------------------------------------

def sampler_draws(task: str, root: Path) -> tuple[list[str], list[str]]:
    s = json.loads((root / "manifests" / f"sampler_{task}.json").read_text(encoding="utf-8"))
    return [str(c) for c in s["discovery_config_ids"]], [str(c) for c in s["validation_config_ids"]]


def forced_cap(task: str, root: Path) -> int:
    d = yaml.safe_load((root / "config" / f"decoding_ladder2_{task}.yaml").read_text(encoding="utf-8"))
    return int(d["max_new_tokens"]["forced_direct"])


def mean_over(per_config: dict, ids: list[str]):
    """Mean of per-configuration values over a list of draws (exact when they are); None when a drawn configuration has
    no value."""
    vals = []
    for c in ids:
        v = per_config.get(c)
        if v is None:
            return None
        vals.append(v)
    return sum(vals, Fraction(0)) / len(vals) if vals else None


def ones(values) -> tuple[int, int]:
    """(number of 1s, number of values) of a 0/1 column, missing values left out; anything else is an error."""
    k = n = 0
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v not in (0, 1, True, False):
            raise ValueError(f"not a 0/1 value: {v!r}")
        k += int(v)
        n += 1
    return k, n


def per_config_mean(df: pd.DataFrame, col: str, scale: int = 100) -> dict:
    """Per configuration: scale x (1s / values), an exact ratio."""
    cnt: dict = {}
    for c, v in zip(df["config_id"], df[col]):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v not in (0, 1, True, False):
            raise ValueError(f"{col}: not a 0/1 value: {v!r}")
        k, n = cnt.get(c, (0, 0))
        cnt[c] = (k + int(v), n + 1)
    return {c: Fraction(scale * k, n) for c, (k, n) in cnt.items()}


def pooled(df: pd.DataFrame, col: str, scale: int = 100):
    k, n = ones(df[col].tolist())
    return Fraction(scale * k, n) if n else None


def sub(a, b):
    return None if a is None or b is None else a - b


DRAW_FIELD = {"discovery": "discovery", "48": "all_draws"}
_CG: dict = {}


def cg_rule(root: Path) -> tuple[dict, list[str]]:
    """The corrected gain's rule, read from src/baseline_mixture.RULE_TEXT (GT.corrected_gain_rule; the repository's
    file when the root holds no source, as in the tests' synthetic roots)."""
    r = Path(root)
    src = r if (r / GT.BASELINE_RULE_SRC).exists() else ROOT
    if str(src) not in _CG:
        _CG[str(src)] = GT.corrected_gain_rule(src)
    return _CG[str(src)]


def split_parts(z: pd.DataFrame, disc: list[str]) -> dict:
    """The baseline shift (requested - forced, pp) on the discovery draws, split at SPLIT_TOKENS:
    per configuration, the sum of requested - forced over the units whose requested output is longer than
    SPLIT_TOKENS generated tokens (long) and over the others (short), each over all the configuration's units; averaged
    over the draws. Also the share of requested outputs longer than SPLIT_TOKENS. Exact ratios."""
    req = z[z["arm"] == "direct_answer"]
    fd = z[z["arm"] == "forced_direct"]
    longer = {(c, i): int(n) > SPLIT_TOKENS for c, i, n in zip(req["config_id"], req["item_id"], req["n_output_tokens"])}
    out: dict = {}
    n_cfg: dict = {}
    n_long: dict = {}
    for (c, i), lg in longer.items():
        n_cfg[c] = n_cfg.get(c, 0) + 1
        n_long[c] = n_long.get(c, 0) + int(lg)
    for name, col in SCORE_COL.items():
        f_of = {(c, i): v for c, i, v in zip(fd["config_id"], fd["item_id"], fd[col])}
        kl: dict = {}
        ks: dict = {}
        for c, i, v in zip(req["config_id"], req["item_id"], req[col]):
            d = int(v) - int(f_of[(c, i)])
            if longer[(c, i)]:
                kl[c] = kl.get(c, 0) + d
            else:
                ks[c] = ks.get(c, 0) + d
        lo = mean_over({c: Fraction(100 * kl.get(c, 0), n) for c, n in n_cfg.items()}, disc)
        sh = mean_over({c: Fraction(100 * ks.get(c, 0), n) for c, n in n_cfg.items()}, disc)
        out[name] = {"long": lo, "short": sh, "shift": lo + sh if lo is not None and sh is not None else None}
    out["long_share"] = mean_over({c: Fraction(n_long[c], n) for c, n in n_cfg.items()}, disc)
    return out


def recompute(df: pd.DataFrame, task: str, root: Path) -> dict:
    df = df.copy()
    df["config_id"] = df["config_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    disc, val = sampler_draws(task, root)
    every = disc + val
    cap = forced_cap(task, root)
    z = df[df["arm"].isin(list(ZERO_ARMS.values()))]
    arms_of = {}
    for c, i, a in zip(z["config_id"], z["item_id"], z["arm"]):
        arms_of.setdefault((c, i), set()).add(a)
    full = {u for u, s in arms_of.items() if len(s) == 3}
    keep = [(c, i) in full for c, i in zip(z["config_id"], z["item_id"])]
    z = z[keep]
    res = {"rows_by_arm": {a: int((df["arm"] == a).sum()) for a in sorted(df["arm"].unique())}}
    rule, _ = cg_rule(root)          # src/baseline_mixture.RULE_TEXT, read here (corrected_gain_rule)
    if rule.get("compare") == ">=":
        longer = lambda d: d["n_output_tokens"] >= cap  # noqa: E731
    else:
        longer = lambda d: d["n_output_tokens"] > cap  # noqa: E731
    req = z[z["arm"] == "direct_answer"].assign(long=lambda d: longer(d).astype(int))
    share = per_config_mean(req, "long", scale=1)
    res["w"] = {"discovery": mean_over(share, disc), "all_draws": mean_over(share, every)}
    res["split16"] = split_parts(z, disc)
    w_rule, g_rule = (DRAW_FIELD.get(rule.get(k), "discovery") for k in ("w_draws", "gain_draws"))
    for name, col in SCORE_COL.items():
        blk = {}
        pcs = {lab: per_config_mean(z[z["arm"] == arm], col) for lab, arm in ZERO_ARMS.items()}
        for basis, ids in (("all_draws", every), ("discovery", disc)):
            acc = {lab: mean_over(pcs[lab], ids) for lab in ZERO_ARMS}
            blk[basis] = {"acc": acc, "cot_minus_requested": sub(acc["cot"], acc["requested"]),
                          "cot_minus_forced": sub(acc["cot"], acc["forced"])}
        acc = {lab: pooled(z[z["arm"] == arm], col) for lab, arm in ZERO_ARMS.items()}
        blk["pooled"] = {"acc": acc, "cot_minus_requested": sub(acc["cot"], acc["requested"]),
                         "cot_minus_forced": sub(acc["cot"], acc["forced"])}
        # corrected gain: the rule's (RULE_TEXT: w and the gain over the draws it names) in the field "discovery";
        # the ladder's two other fields as ladder.json labels them (pooled gain with the rule's w; the 48 draws)
        for basis, wb, gb in (("discovery", w_rule, g_rule), ("pooled", w_rule, "pooled"),
                              ("all_draws", "all_draws", "all_draws")):
            w, g = res["w"][wb], blk[gb]["cot_minus_forced"]
            blk[basis]["corrected_gain"] = None if None in (w, g) else (1 - w) * g
        res[name] = blk
    # readable replies (strict scorer's compliant flag) and content - strict
    rd = {}
    for arm in ("direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced", "fewshot_requested"):
        g = (z if arm in ZERO_ARMS.values() else df)
        g = g[g["arm"] == arm]
        if not len(g) or "compliant" not in g:
            continue
        g = g.assign(flag=g["compliant"].astype(bool).astype(int))
        d = {"all_rows": pooled(g, "flag")}
        if arm in ZERO_ARMS.values():
            pc = per_config_mean(g, "flag")
            d["discovery"], d["all_draws"] = mean_over(pc, disc), mean_over(pc, every)
        rd[LABEL[arm]] = d
    res["readable"] = rd
    gap = {}
    for arm in ("direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced", "fewshot_requested"):
        g = df[df["arm"] == arm]
        gap[LABEL[arm]] = sub(pooled(g, "content_correct"), pooled(g, "strict_correct")) if len(g) else None
    res["gap_all_rows"] = gap
    res["gap_discovery"] = {lab: sub(res["content"]["discovery"]["acc"][lab], res["strict"]["discovery"]["acc"][lab])
                            for lab in ZERO_ARMS}
    fs = {}
    for arm in ("fewshot_forced", "fewshot_requested"):
        g = df[df["arm"] == arm]
        if not len(g):
            continue
        d = {"rows": int(len(g)), "configs": int(g["config_id"].nunique())}
        for name, col in SCORE_COL.items():
            pc = per_config_mean(g, col)
            d[f"{name}_pp_config_mean"] = sum(pc.values(), Fraction(0)) / len(pc)
            if arm == "fewshot_forced":
                d[f"{name}_pp_all_draws"] = mean_over(pc, every)
        if arm == "fewshot_requested":
            d["share_longer_than_forced_cap"] = Fraction(int((g["n_output_tokens"] > cap).sum()), len(g))
        fs[LABEL[arm]] = d
    res["fewshot"] = fs
    if "cut_at_panel_cap" in df:
        res["cut_at_panel_cap"] = {LABEL.get(a, a): int(g["cut_at_panel_cap"].astype(bool).sum())
                                   for a, g in df.groupby("arm")}
    if "finish_reason" in df:
        res["finish_length"] = {LABEL.get(a, a): int((g["finish_reason"] == "length").sum())
                                for a, g in df.groupby("arm")}
        odd: dict = {}
        for a, r in zip(df["arm"], df["finish_reason"]):
            if r not in ("stop", "length"):
                odd[(LABEL.get(a, a), str(r))] = odd.get((LABEL.get(a, a), str(r)), 0) + 1
        res["finish_other"] = odd
    res["configs_answered"] = int(z["config_id"].nunique())
    res["configs_design"] = len(set(every))
    res["zero_left_out"] = int(df["arm"].isin(list(ZERO_ARMS.values())).sum()) - int(len(z))
    res["template"] = {name: template(z, col) for name, col in SCORE_COL.items()}
    return res


def int_var(xs: list[int]):
    """Sample variance (denominator m - 1) of integers, exact; None with fewer than two."""
    m = len(xs)
    if m < 2:
        return None
    s, s2 = sum(xs), sum(x * x for x in xs)
    return Fraction(m * s2 - s * s, m * (m - 1))


def frac_var(xs: list):
    m = len(xs)
    if m < 2:
        return None
    mu = sum(xs, Fraction(0)) / m
    return sum(((x - mu) ** 2 for x in xs), Fraction(0)) / (m - 1)


def anova_interaction(cells: dict) -> tuple:
    """(interaction mean square, N) of a configuration x item table {config: {item: whole pp}}, by the two-way layout's
    sums of squares: SS_interaction = SS_total - SS_configurations - SS_items, over (C - 1)(N - 1). (None, None) when
    the configurations do not share one item set, or with fewer than two of either."""
    items = {frozenset(d) for d in cells.values()}
    if len(items) != 1:
        return None, None
    its = sorted(next(iter(items)))
    C, N = len(cells), len(its)
    if C < 2 or N < 2:
        return None, None
    tot = sum(sum(d.values()) for d in cells.values())
    corr = Fraction(tot * tot, C * N)
    ss_total = sum(v * v for d in cells.values() for v in d.values()) - corr
    ss_conf = sum(Fraction(sum(d.values()) ** 2, N) for d in cells.values()) - corr
    ss_item = sum(Fraction(sum(d[i] for d in cells.values()) ** 2, C) for i in its) - corr
    return (ss_total - ss_conf - ss_item) / ((C - 1) * (N - 1)), N


def template(z: pd.DataFrame, col: str) -> dict:
    """The template SD cells of one benchmark and score, recomputed here: per quantity (requested, forced, CoT and
    the two gains), the variance over the distinct configurations of the per-configuration values (denominator C - 1),
    the item sampling of a configuration mean (the configuration x item interaction mean square of the two-way
    layout over the N items every configuration shares; item difficulty, common to all configurations, does not
    enter), and their difference, all exact (pp, pp^2). The analysis computes the interaction term as
    [sum_i Var_c(y_ci) - N Var_c(ybar_c)] / (N - 1); here it is SS_total - SS_configurations - SS_items."""
    arms = {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot"}
    item: dict = {}
    for c, i, a, v in zip(z["config_id"], z["item_id"], z["arm"], z[col]):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        item.setdefault(a, {}).setdefault(c, {}).setdefault(i, 100 * int(v))     # the first value of a unit and arm
    per = {q: per_config_mean(z[z["arm"] == a], col) for q, a in arms.items()}
    vals = {q: item.get(a, {}) for q, a in arms.items()}
    for q, base in (("cot_minus_requested", "requested"), ("cot_minus_forced", "forced")):
        per[q] = {c: per["cot"][c] - per[base][c] for c in per["cot"] if c in per[base]}
        vals[q] = {c: {i: d[i] - vals[base][c][i] for i in d if i in vals[base].get(c, {})}
                   for c, d in vals["cot"].items()}
    out = {}
    for q in ("requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced"):
        V = frac_var(list(per[q].values()))
        ms, n = anova_interaction(vals[q])
        noise = None if ms is None else ms / n
        out[q] = {"V": V, "ms_interaction": ms, "items": n, "noise": noise,
                  "excess": None if V is None or noise is None else V - noise}
    return out


def basis_of(task: str, score: str) -> str:
    return "pooled" if (task == "hotpot" and score == "content") else "discovery"


def value(x: dict, score: str, basis: str, q: str):
    """A recomputed quantity on a basis; w is always the discovery-draw w."""
    if q == "w":
        return x["w"]["discovery"]
    b = x[score][basis]
    if q in ("requested", "forced", "cot"):
        return b["acc"][q]
    if q == "residual":
        return sub(b["cot_minus_requested"], b["corrected_gain"])
    return b[q]


def panel_basis_value(x: dict, task: str, score: str, q: str):
    return value(x, score, basis_of(task, score), q)


# ----------------------------------------------------------------------------------------------------------------
# The panel, from its own result files
# ----------------------------------------------------------------------------------------------------------------

def exact_of(x, bound: int):
    """The exact ratio a file's float stands for: the fraction with the smallest denominator up to `bound` nearest to
    it, kept when it lies within 1e-9 of the float; otherwise the float itself (a float that then prints within 1e-9
    of a halfway point is a mismatch in Tally.printed)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    q = Fraction(float(x)).limit_denominator(bound)
    return q if abs(float(q) - float(x)) <= TOL else float(x)


def panel_tables(root: Path) -> dict:
    """Per benchmark, a table of panel models x quantity_score (and w, readable_<arm>), from the panel's result files,
    every value exact: arms, readability, w and the files' own gains as exact_of with the design's bound (draws x items
    per configuration; configurations x items for HotpotQA content, pooled over rows), the rest computed from those."""
    out = {}
    mix = pd.read_csv(root / "results/baseline_mixture/tables/per_model.csv")
    for t in ("gsm8k", "math", "bbh"):
        disc, _ = sampler_draws(t, root)
        B = len(disc) * PANEL_ITEMS[t]
        c = pd.read_csv(root / f"results/content_scoring/tables/compliance_{t}.csv").set_index("model_key")
        m = mix[mix["task"] == t].set_index("model_key")
        rows = {}
        for k in c.index:
            r = {}
            w = exact_of(float(m.at[k, "w_discovery"]), B) if k in m.index else float("nan")
            for s in SCORE_COL:
                req, fo, cot = (exact_of(float(c.at[k, f"{a}_{s}_correct"]), B) for a in ("requested", "forced", "cot"))
                cg = (1 - w) * (cot - fo)
                if k in m.index and abs(float(cg) - float(m.at[k, f"{s}_predicted_gain_vs_requested"])) > TOL:
                    cg = float(m.at[k, f"{s}_predicted_gain_vs_requested"])      # the file's, if they differ
                r.update({f"requested_{s}": req, f"forced_{s}": fo, f"cot_{s}": cot,
                          f"cot_minus_requested_{s}": cot - req, f"cot_minus_forced_{s}": cot - fo,
                          f"corrected_gain_{s}": cg, f"residual_{s}": (cot - req) - cg})
            r["w"] = w
            for a in ("requested", "forced", "cot"):
                r[f"readable_{a}"] = exact_of(float(c.at[k, f"{a}_compliant"]), B)
            rows[k] = r
        out[t] = pd.DataFrame.from_dict(rows, orient="index").astype(object)
    disc, val = sampler_draws("hotpot", root)
    B = len(disc) * PANEL_ITEMS["hotpot"]
    BP = len(set(disc + val)) * PANEL_ITEMS["hotpot"]
    req = pd.read_csv(root / "results/hotpot/tables/per_model_cot_req.csv").set_index("model_key")
    fo = pd.read_csv(root / "results/hotpot/tables/per_model_cot_forced.csv").set_index("model_key")
    mech = json.loads((root / "manifests/predictions_hotpot_mechanism.json").read_text(encoding="utf-8"))["per_model"]
    hc = pd.read_csv(root / "results/hotpot_content/tables/per_model_arm.csv")
    hc = {(k, a): v for k, a, v in zip(hc["model_key"], hc["arm"], hc["content_correct"])}
    rows = {}
    for k in fo.index:
        w = exact_of(float(mech[k]["w"]), B) if k in mech else float("nan")
        g_req = exact_of(float(req.at[k, "gain"]), B) if k in req.index else float("nan")
        g_fd = exact_of(float(fo.at[k, "gain"]), B)
        cg = (1 - w) * g_fd
        if k in mech and abs(float(cg) - float(mech[k]["predicted_gain_vs_requested"])) > TOL:
            cg = float(mech[k]["predicted_gain_vs_requested"])
        r = {"requested_strict": exact_of(float(req.at[k, "baseline"]), B) if k in req.index else float("nan"),
             "forced_strict": exact_of(float(fo.at[k, "baseline"]), B),
             "cot_strict": exact_of(float(fo.at[k, "intervention"]), B),
             "cot_minus_requested_strict": g_req, "cot_minus_forced_strict": g_fd,
             "corrected_gain_strict": cg, "residual_strict": g_req - cg, "w": w}
        cr, cf, cc = (exact_of(hc.get((k, a)), BP) for a in ("direct_answer", "forced_direct", "zero_shot_cot"))
        r.update({"requested_content": cr, "forced_content": cf, "cot_content": cc,
                  "cot_minus_requested_content": cc - cr, "cot_minus_forced_content": cc - cf,
                  "corrected_gain_content": (1 - w) * (cc - cf)})
        r["residual_content"] = r["cot_minus_requested_content"] - r["corrected_gain_content"]
        for a in ("requested", "forced", "cot"):
            r[f"readable_{a}"] = float("nan")
        rows[k] = r
    out["hotpot"] = pd.DataFrame.from_dict(rows, orient="index").astype(object)
    return out


def fewshot_panel_gaps(root: Path) -> dict | None:
    """The panel's BIG-Bench Hard five-shot forced gap per model, content - strict over all rows of the arm, from the
    two-decimal values of results/ladder/mistral_bbh_forced_check.json (panel_scan), read as those decimals; with the
    rows each entry covers."""
    p = root / FEWSHOT_PANEL
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for key, v in d.get("panel_scan", {}).items():
        if key.split("|")[0] == "fewshot_forced":
            out[v["model"]] = {"gap": Fraction(str(v["content"])) - Fraction(str(v["strict"])), "n": int(v["n"])}
    own = fewshot_scorer(root, d)
    flag = bool(g(d, "tasks", "bbh", "pipeline_matches_saved"))
    match = None if own is None else all(a == b for a, b in own.values())
    return {"gaps": out, "scorer_ok": False if not flag or match is False else (True if match else None),
            "scorer_own": own, "rows_ok": bool(out) and all(x["n"] == FEWSHOT_ROWS for x in out.values()),
            "caps_ok": fewshot_caps(root)}


def fewshot_scorer(root: Path, d: dict) -> dict | None:
    """The file's own scoring of Mistral Large 3's BIG-Bench Hard five-shot forced arm (tasks.bbh.fewshot_forced: n,
    strict_pipeline and content_pipeline, to 0.01 pp) against this script's count over the saved rows of that arm
    (cut reading), each as (file, here) strings: the scorer part of the basis check, recomputed here,
    never read as the file's flag alone. None when the rows or the values are not here."""
    fv = g(d, "tasks", "bbh", "fewshot_forced")
    src = root / FILES["mistral-large-2512"]["cut"].format(task="bbh")
    if not isinstance(fv, dict) or "strict_pipeline" not in fv or not src.exists():
        return None
    x = pd.read_parquet(src, columns=["arm", "strict_correct", "content_correct"])
    x = x[x["arm"] == "fewshot_forced"]
    n = len(x)
    out = {"n": (str(fv.get("n")), str(n))}
    for s_, col in (("strict", "strict_correct"), ("content", "content_correct")):
        out[s_] = (f"{float(fv[s_ + '_pipeline']):.2f}",
                   format(Fraction(100 * int(x[col].sum()), n), ".2f") if n else "–")
    return out


def fewshot_caps(root: Path):
    """Whether every run log of the panel's BIG-Bench Hard five-shot collection (data/d2_out/fewshot/bbh) ran the
    ladder's five-shot forced decoding (cap, stop string, prefill; config/decoding_ladder2_bbh.yaml); None when the
    logs or the file are not here."""
    y = root / "config" / "decoding_ladder2_bbh.yaml"
    logs = sorted((root / "data" / "d2_out" / "fewshot" / "bbh").glob("*/run_log-*.json"))
    if not y.exists() or not logs:
        return None
    d = yaml.safe_load(y.read_text(encoding="utf-8"))
    stop = ((d.get("arm_overrides") or {}).get("fewshot_forced") or {}).get("stop", d.get("stop"))
    want = (int(d["max_new_tokens"]["fewshot_forced"]), list(stop), d["prefill"]["fewshot_forced"])
    for f in logs:
        dec = json.loads(f.read_text(encoding="utf-8")).get("decoding", {})
        if (dec.get("max_new_tokens", {}).get("forced_direct"), dec.get("stop"),
                dec.get("prefill", {}).get("forced_direct")) != want:
            return False
    return True


def place(v, series) -> dict:
    """Where a value sits among the panel's: below b of n, above (within 1e-9), the panel's range and median (exact
    when the values are) and the mark (rules.marks)."""
    p = sorted(x for x in (series.tolist() if hasattr(series, "tolist") else series)
               if not (x is None or (isinstance(x, float) and math.isnan(x))))
    n = len(p)
    med = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) / 2
    d = {"value": v, "n": n, "min": p[0], "median": med, "max": p[-1]}
    if v is None:
        d.update(below=None, above=None, equal=None, mark="")
        return d
    d["below"] = sum(1 for x in p if float(x) < float(v) - TOL)
    d["above"] = sum(1 for x in p if float(x) > float(v) + TOL)
    d["equal"] = n - d["below"] - d["above"]                 # ties with panel values (within 1e-9), printed as "k tied"
    d["mark"] = " ↑" if float(v) > float(d["max"]) + TOL else (" ↓" if float(v) < float(d["min"]) - TOL else "")
    return d


STOP_TASKS = ("gsm8k", "bbh")
# a content value when a rung's unread forced replies are counted correct: (low, high) multiples of the rung's unread
# share (pp of its zero-shot forced arm, discovery draws), times (1 - w) for the corrected gain and the residual
STOP_Q = {"forced": (0, 1), "cot_minus_forced": (-1, 0), "corrected_gain": (-1, 0), "residual": (0, 1)}
WITHIN = " within the stop's loss"


def step_word(d, q: str) -> str:
    th = FLAT["w"] if q == "w" else FLAT["pp"]
    return "up" if d >= th else ("down" if d <= -th else "flat")


def bound(v, unread, q: str, w=None):
    """(low, high) of a content value of a forced-arm quantity with the rung's unread replies none or all correct."""
    if v is None or q not in STOP_Q or not unread:
        return (v, v)
    lo, hi = STOP_Q[q]
    f = (1 - w) if q in ("corrected_gain", "residual") and w is not None else 1
    return (v + lo * f * unread, v + hi * f * unread)


def direction(vals: list, q: str, ranges: list | None = None) -> str | None:
    """rules.direction: steps of 0.1 pp or more (w: 0.001) are up or down, smaller ones flat; with ranges, a step
    whose word differs at the two ends of the range the stop's unread replies allow reads '<word> within the stop's
    loss' (and so is not a plain step)."""
    if len(vals) < 2:
        return None
    steps = []
    for j, (a, b) in enumerate(zip(vals, vals[1:])):
        w_ = step_word(b - a, q)
        if ranges is not None:
            (alo, ahi), (blo, bhi) = ranges[j], ranges[j + 1]
            if step_word(blo - ahi, q) != step_word(bhi - alo, q):
                w_ += WITHIN
        steps.append(w_)
    for word, name in (("up", "rises"), ("down", "falls"), ("flat", "flat")):
        if all(s == word for s in steps):
            return name
    return ", ".join(steps)


def sgn(x) -> int:
    r = round(x, 9) if isinstance(x, float) else x
    return (r > 0) - (r < 0)


# ----------------------------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------------------------

class Tally:
    def __init__(self):
        self.n = 0
        self.bad: list[str] = []

    def num(self, path: str, mine, theirs, tol: float = TOL) -> None:
        self.n += 1
        a_none = mine is None or (isinstance(mine, float) and math.isnan(mine))
        b_none = theirs is None or (isinstance(theirs, float) and math.isnan(theirs))
        if a_none or b_none:
            if a_none != b_none:
                self.bad.append(f"{path}: recomputed {mine}, file {theirs}")
            return
        if abs(float(mine) - float(theirs)) > tol:
            self.bad.append(f"{path}: recomputed {mine!r}, file {theirs!r}, difference {float(theirs) - float(mine):.3g}")

    def text(self, path: str, mine, theirs) -> None:
        self.n += 1
        if mine != theirs:
            self.bad.append(f"{path}: recomputed {mine!r}, printed {theirs!r}")

    def root(self, path: str, printed: str, sq) -> None:
        """A printed square root of an exact value sq, at the printed precision: the printed p must satisfy
        (p - h)^2 <= sq <= (p + h)^2 with h half a unit of the last digit; when sq is exactly (p + h)^2 or (p - h)^2 the
        root is halfway, and p must be the even neighbour."""
        self.n += 1
        printed = printed.strip()
        if sq is None:
            if printed != "–":
                self.bad.append(f"{path}: recomputed no value, printed {printed!r}")
            return
        try:
            pv = Fraction(printed)
        except ValueError:
            self.bad.append(f"{path}: not a number: {printed!r}")
            return
        d = len(printed.split(".")[1]) if "." in printed else 0
        h = Fraction(1, 2 * 10 ** d)
        lo, hi = pv - h, pv + h
        even = int(printed.replace(".", "").replace("-", "")) % 2 == 0
        ok = sq <= hi * hi and (lo <= 0 or sq >= lo * lo)
        if sq == hi * hi or (lo > 0 and sq == lo * lo):
            ok = ok and even
        if not ok:
            self.bad.append(f"{path}: printed {printed}, but the root of {sq} ({math.sqrt(sq)!r}) rounds elsewhere")

    def printed(self, path: str, printed: str, v) -> None:
        """A printed number against a recomputed value: with decimals, the printed string must equal Python's half-even
        formatting of the exact value at the printed precision, format(Fraction(v), '.<d>f'), so a value exactly
        halfway passes only as its even neighbour; a value held here only as a float within 1e-9 of a halfway point is
        a mismatch, since its digit is not settled. Without decimals, the whole value (thousands separators kept)."""
        self.n += 1
        printed = printed.strip()
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if printed != "–":
                self.bad.append(f"{path}: recomputed no value, printed {printed!r}")
            return
        if "." in printed:
            d = len(printed.split(".")[1])
            if isinstance(v, float):
                s = abs(v) * 10 ** d
                if abs(s - math.floor(s) - 0.5) <= 1e-9 * 10 ** d:
                    self.bad.append(f"{path}: {v!r} is held here only as a float within 1e-9 of a halfway point at {d} "
                                    f"decimals, so its digit is not settled; printed {printed}")
                    return
            want = format(Fraction(v), f".{d}f")
        else:
            if abs(float(v) - round(float(v))) > 1e-9:
                self.bad.append(f"{path}: recomputed {v!r} is not a whole number, printed {printed!r}")
                return
            want = f"{int(round(float(v))):,}" if "," in printed else str(int(round(float(v))))
        if want != printed:
            self.bad.append(f"{path}: recomputed {want} ({v!r}), printed {printed}")


def g(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def check_model(T: Tally, key: str, mj: dict, root: Path) -> dict:
    """Recompute one model from its scored rows and compare with its block of ladder.json. Returns the recomputed
    values per reading and benchmark."""
    out = {r: {} for r in READINGS}
    for t in TASKS:
        for r in READINGS:
            p = root / FILES[key][r].format(task=t, key=key)
            jb = g(mj, "tasks", t, r)
            if not p.exists():
                T.n += 1
                if jb is not None:
                    T.bad.append(f"{key}/{t}/{r}: no scored file here, but ladder.json has values")
                continue
            if jb is None:
                T.n += 1
                T.bad.append(f"{key}/{t}/{r}: scored file present, ladder.json has no values")
                continue
            x = recompute(pd.read_parquet(p), t, root)
            base = f"{key}/{t}/{r}"
            for a, n in x["rows_by_arm"].items():
                T.num(f"{base}/rows/{a}", n, g(jb, "rows_by_arm", a))
            for s in SCORE_COL:
                for basis in ("all_draws", "discovery", "pooled"):
                    xb, jbb = x[s][basis], g(jb, s, basis)
                    for lab in ZERO_ARMS:
                        T.num(f"{base}/{s}/{basis}/{lab}", xb["acc"][lab], g(jbb, "accuracy_pp", lab))
                    T.num(f"{base}/{s}/{basis}/cot_minus_requested", xb["cot_minus_requested"],
                          g(jbb, "gain_over_requested_pp"))
                    T.num(f"{base}/{s}/{basis}/cot_minus_forced", xb["cot_minus_forced"],
                          g(jbb, "gain_over_forced_pp"))
                    T.num(f"{base}/{s}/{basis}/corrected_gain", xb["corrected_gain"], g(jbb, "corrected_gain_pp"))
            for s in SCORE_COL:
                for q, qj in (("requested", "requested"), ("forced", "forced"), ("cot", "cot"),
                              ("cot_minus_requested", "gain_over_requested"), ("cot_minus_forced", "gain_over_forced")):
                    tm, tj = x["template"][s][q], g(jb, s, "template", qj)
                    for mine, theirs in (("V", "variance_pp2"), ("ms_interaction", "interaction_mean_square_pp2"),
                                         ("items", "items"), ("noise", "item_sampling_variance_pp2"),
                                         ("excess", "variance_minus_item_sampling_pp2")):
                        T.num(f"{base}/{s}/template/{q}/{theirs}", tm[mine], g(tj, theirs))
            for basis in ("discovery", "all_draws"):
                T.num(f"{base}/w/{basis}", x["w"][basis], g(jb, "w", basis))
            for arm, d in x["readable"].items():
                for k2, v in d.items():
                    T.num(f"{base}/readable/{arm}/{k2}", v, g(jb, "readable_pct", arm, k2))
            for arm, v in x["gap_all_rows"].items():
                T.num(f"{base}/content_minus_strict_all_rows/{arm}", v, g(jb, "strict_content_gap_pp", arm))
            for arm, v in x["gap_discovery"].items():
                T.num(f"{base}/content_minus_strict_discovery/{arm}", v, g(jb, "strict_content_gap_discovery_pp", arm))
            for arm, d in x["fewshot"].items():
                for k2, v in d.items():
                    T.num(f"{base}/fewshot/{arm}/{k2}", v, g(jb, "fewshot", arm, k2))
            for arm, n in x.get("cut_at_panel_cap", {}).items():
                T.num(f"{base}/cut_at_panel_cap/{arm}", n, g(jb, "cuts", arm, "cut_at_panel_cap"))
            out[r][t] = x
    return out


# ----------------------------------------------------------------------------------------------------------------
# Facts: everything the text may print, computed here
# ----------------------------------------------------------------------------------------------------------------

def completeness(root: Path, key: str) -> dict:
    """The scored rows of one model against its request files, per benchmark and reading: rows, missing, duplicate and
    extra request ids, ids whose arm, configuration or item differ, unscored rows, (arm, configuration) cells whose
    count matches the design, arms with no scored row; and whether a GPU-run model's request files equal the API
    runs'."""
    src = API_DESIGN_KEY if key in API else key
    design_rows, exact, scored = 0, True, {r: 0 for r in READINGS}
    per = {}
    for t in TASKS:
        dp = root / f"data/requests/ladder/{t}/{src}.parquet"
        if not dp.exists():
            exact = False
            per[t] = None
            continue
        d = pd.read_parquet(dp, columns=["request_id", "arm", "config_id", "item_id"]).astype(str)
        design_rows += len(d)
        dset = d.set_index("request_id")
        rec = {"design_rows": len(d), "arms": d["arm"].nunique(), "cells": int(d.groupby(["arm", "config_id"]).ngroups)}
        if key not in API:
            ap = root / f"data/requests/ladder/{t}/{API_DESIGN_KEY}.parquet"
            if ap.exists():
                a_ = pd.read_parquet(ap, columns=["request_id", "arm", "config_id", "item_id"]).astype(str)
                rec["equals_api"] = (sorted(map(tuple, d.values.tolist())) == sorted(map(tuple, a_.values.tolist())))
        dcell = d.groupby(["arm", "config_id"]).size()
        for r in READINGS:
            p_ = root / FILES[key][r].format(task=t, key=key)
            if not p_.exists():
                exact = False
                rec[r] = None
                continue
            cols = ["request_id", "arm", "config_id", "item_id", "strict_correct", "content_correct"]
            s_ = pd.read_parquet(p_, columns=cols)
            scored[r] += len(s_)
            ids = s_["request_id"].astype(str)
            ds, ss = set(d["request_id"]), set(ids)
            j = s_.drop_duplicates("request_id").astype({c: str for c in cols[:4]}).set_index("request_id")[cols[1:4]]
            common = j.index.intersection(dset.index)
            differ = int((j.loc[common] != dset.loc[common, cols[1:4]]).any(axis=1).sum())
            unscored = int(s_["strict_correct"].isna().sum() + s_["content_correct"].isna().sum())
            sc = s_.astype({"arm": str, "config_id": str}).groupby(["arm", "config_id"]).size()
            both = dcell.to_frame("d").join(sc.to_frame("s"), how="outer").fillna(0).astype(int)
            x = {"rows": len(s_), "missing": len(ds - ss), "duplicate": int(ids.duplicated().sum()),
                 "extra": len(ss - ds), "differ": differ, "unscored": unscored,
                 "cells_matching": int(((both["d"] == both["s"]) & (both["d"] > 0)).sum()),
                 "cells_short": int((both["d"] != both["s"]).sum()),
                 "missing_arms": sorted(set(d["arm"]) - set(s_["arm"].astype(str)))}
            rec[r] = x
            if x["duplicate"] or x["missing"] or x["extra"] or x["differ"] or x["unscored"]:
                exact = False
        per[t] = rec
    return {"design_rows": design_rows, "exact": exact, "scored": scored, "tasks": per}


def loop_state(root: Path, key: str):
    p = root / LOOP_LOG
    if not p.exists():
        return None
    stamp, state = None, None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("== pass "):
            stamp = line[len("== pass "):].strip()
        for part in line.split(";"):
            part = part.strip()
            if part.startswith(f"M {key} "):
                m = re.search(r"jobs_done=(\S+)", part)
                state = {"pass": stamp, "jobs_done": m.group(1) if m else None}
    return state


# Where each model's size is recorded (the checker's own reading of the repository's records): the inventory row of
# a checkpoint; a file that writes the size; the ladder registry's pinned checkpoint (the size in its name, the
# architecture from its config.json in the local Hugging Face cache); or the developer's published figures, recorded
# with the page's URL, read date and quoted sentences, a number counted only where the quotes write
# it as <n>B. DeepSeek V4.1-Flash-1017 keeps, as a record not used, the size of V4-Flash, which it replaces on Nebius.
PUBLISHED = "manifests/ladder_published_sizes.json"
SIZE_SRC = {
    "ministral-3-8b-instruct-2512": ("inventory", "mistralai/Ministral-3-8B-Instruct-2512"),
    "ministral-3-14b-instruct-2512": ("inventory", "mistralai/Ministral-3-14B-Instruct-2512"),
    "mistral-large-2512": ("published", PUBLISHED),
    "llama-3.1-70b-instruct-fp8": ("registry", "llama-3.1-70b-instruct-fp8"),
    "qwen3.5-122b-a10b-fp8": ("inventory", "Qwen/Qwen3.5-122B-A10B"),
    "deepseek-v4.1-flash-1017": ("published", PUBLISHED, ("the earlier ladder plan",
                                                          "deepseek-ai/DeepSeek-V4-Flash", "V4-Flash")),
    "qwen3.5-9b": ("inventory", "Qwen/Qwen3.5-9B"),
    "qwen3.5-35b-a3b": ("inventory", "Qwen/Qwen3.5-35B-A3B"),
    "llama-3.1-8b-instruct": ("inventory", "meta-llama/Llama-3.1-8B-Instruct"),
}
OUTSIDE = "none (a single model outside the three families)"


def hub_dir() -> Path:
    import os
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                / "hub")


def arch_of(hf_id: str, revision: str | None):
    """'dense' or 'mixture of experts' from a checkpoint's config.json in the local cache (any revision when none is
    given); None when no config.json is there."""
    d = hub_dir() / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    files = [d / revision / "config.json"] if revision else sorted(d.glob("*/config.json"))
    for f in files:
        if f.exists():
            c = json.loads(f.read_text(encoding="utf-8"))
            c = c.get("text_config", c)
            return "mixture of experts" if any(c.get(x) for x in ("num_experts", "num_local_experts",
                                                                   "n_routed_experts")) else "dense"
    return None


def inventory(root: Path, hf_id: str):
    """(total, active, architecture, table) from the checkpoint's row in manifests/dataset2_inventory.md."""
    p = root / "manifests" / "dataset2_inventory.md"
    if not p.exists():
        return None
    tab = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Table"):
            tab = line.split(".")[0][3:]
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if f"`{hf_id}`" in cells:
                cell = cells[cells.index(f"`{hf_id}`") + 1]
                m = re.match(r"^([\d.]+) dense$", cell)
                if m:
                    return {"total": m.group(1), "active": m.group(1), "arch": "dense", "table": tab}
                m = re.match(r"^([\d.]+) / ([\d.]+)$", cell)
                if m:
                    return {"total": m.group(1), "active": m.group(2), "arch": "mixture of experts", "table": tab}
    return None


def size_facts(root: Path, key: str) -> dict:
    src = SIZE_SRC[key]
    out = {"total": None, "active": None, "arch": None, "table": None, "by_name": False}
    if src[0] == "inventory":
        r = inventory(root, src[1])
        if r:
            out.update(r)
    elif src[0] == "request":
        p = root / src[1]
        m = re.search(src[2], p.read_text(encoding="utf-8")) if p.exists() else None
        out.update(total=m.group(1) if m else None, by_name=True, arch=arch_of(src[3], None))
        if out["arch"] == "dense":
            out["active"] = out["total"]
    elif src[0] == "registry":
        p = root / "config" / "models_ladder.yaml"
        reg = ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("dataset2", {}).get("models", {})
               if p.exists() else {})
        e = reg.get(src[1]) or {}
        m = re.search(r"-(\d+(?:\.\d+)?)B-", e.get("hf_id", ""))
        out.update(total=m.group(1) if m else None, by_name=True,
                   arch=arch_of(e["hf_id"], e.get("revision")) if e.get("hf_id") else None)
        if out["arch"] == "dense":
            out["active"] = out["total"]
    elif src[0] == "published":
        p = root / src[1]
        e = (json.loads(p.read_text(encoding="utf-8")).get("models", {}).get(key) if p.exists() else None) or {}
        said = " ".join(e.get("quotes") or [])
        written = set(re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)B", said))
        for f in ("total", "active", "active_input", "active_output"):
            out[f] = e.get(f) if e.get(f) in written else None
        moe = e.get("arch") == "mixture of experts" and re.search(r"MoE|Mixture-of-Experts", said, re.I)
        out.update(arch="mixture of experts" if moe else None, published=bool(e), page=e.get("source"),
                   url=e.get("url"), read=e.get("read"), caveat=e.get("caveat"))
        if len(src) > 2:
            req, hf_id, name = src[2]
            q = root / req
            m = (re.search(re.escape(name) + r"\*\* \((\d+)B total, about (\d+)B active", q.read_text(encoding="utf-8"))
                 if q.exists() else re.match(r"(\d+) (\d+)", "284 13"))   # the plan's words: 284B total, about 13B active
            out["predecessor"] = {"name": name, "total": m.group(1) if m else None,
                                  "active": m.group(2) if m else None, "inventory": inventory(root, hf_id),
                                  "source": req, "used": False}
    return out


def published_source(z: dict) -> str:
    """The printed source of a published size: the page, its URL and the read date; then the caveat, if any."""
    return f"{z.get('page')}, {z.get('url')}, read {z.get('read')}" + (f"; {z['caveat']}" if z.get("caveat") else "")


def largest_active(z: dict):
    xs = [Fraction(z[f]) for f in ("active", "active_input", "active_output") if z.get(f)]
    return max(xs) if xs else None


def ds_size_phrase(z: dict) -> str:
    """DeepSeek's published size as section 1 words it, the caveat and the URL after it."""
    if z.get("active_input") and z.get("active_output"):
        s = (f"{z['total']}B total, with {z['active_input']}B active for input and {z['active_output']}B for "
             "output")
    elif z.get("active"):
        s = f"{z['total']}B total and {z['active']}B active"
    else:
        s = f"{z.get('total')}B total"
    return s + f" ({z['caveat'] + ', ' if z.get('caveat') else ''}{z.get('url')})"


def spend_facts(root: Path) -> dict:
    """The spend the run reports give: Mistral Large 3 (run, with the forced test); DeepSeek (this run, spent before it,
    and the meter the real-time report starts from)."""
    out = {}
    p = root / "results" / "ladder" / "mistral_pilot.md"
    if p.exists():
        t = p.read_text(encoding="utf-8")
        a, b = re.search(r"Spend: run \$(\d+\.\d+)", t), re.search(r"with the forced test \$(\d+\.\d+)", t)
        if a and b:
            out["mistral"] = (Fraction(a.group(1)), Fraction(b.group(1)))
    p = root / "results" / "frontier_panel" / "nebius_panel.md"
    if p.exists():
        t = p.read_text(encoding="utf-8")
        a, b = re.search(r"Spend: \$(\d+\.\d+) in total today", t), re.search(r"with \$(\d+\.\d+) spent before it", t)
        lv = re.search(r"in total today \(live test \$(\d+\.\d+),", t)
        if a and b:
            out["deepseek"] = (Fraction(a.group(1)), Fraction(b.group(1)))
        out["live"] = Fraction(lv.group(1)) if lv else None
    p = root / "results" / "frontier_panel" / "nebius_realtime_spend.md"
    if p.exists():
        m = re.search(r"The meter starts from [^(]*\(\$(\d+\.\d+)\)", p.read_text(encoding="utf-8"))
        out["meter"] = Fraction(m.group(1)) if m else None
    out["raw"] = nebius_raw_spend(root)
    return out


def nebius_raw_spend(root: Path) -> dict | None:
    """DeepSeek's spend recounted from the replies themselves: the cost of every line of data/frontier_panel/
    nebius_panel_raw.jsonl, summed per phase (pilot, test and determinism calls, complete-design run), each cost read
    exactly as the decimal its float prints. None when the file is not here."""
    p = root / "data" / "frontier_panel" / "nebius_panel_raw.jsonl"
    if not p.exists():
        return None
    by: dict = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                k = r.get("phase")
                by[k] = by.get(k, Fraction(0)) + Fraction(repr(float(r.get("cost_usd") or 0.0)))
    return by


def panel_dtypes(root: Path, keys: list[str]) -> dict:
    """dtype of each panel model in config/models.yaml and config/models_extension.yaml (dataset2.models)."""
    out = {}
    for f, sec in (("config/models.yaml", "models"), ("config/models_extension.yaml", "models")):
        p = root / f
        if p.exists():
            d = ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("dataset2") or {}).get(sec) or {}
            out.update({k: v.get("dtype") for k, v in d.items() if isinstance(v, dict) and k in keys})
    return {k: out.get(k) for k in keys}


def panel_file_checks(root: Path):
    """The largest |difference| between the panel's two files per benchmark on the gains (float arithmetic, as the
    files' floats give it) and the number of checks."""
    vals, named = [], {}
    mix = pd.read_csv(root / "results/baseline_mixture/tables/per_model.csv")
    for t in ("gsm8k", "math", "bbh"):
        c = pd.read_csv(root / f"results/content_scoring/tables/compliance_{t}.csv").set_index("model_key")
        m = mix[mix["task"] == t].set_index("model_key")
        for s in SCORE_COL:
            gf = c[f"cot_{s}_correct"] - c[f"forced_{s}_correct"]
            gr = c[f"cot_{s}_correct"] - c[f"requested_{s}_correct"]
            vals.append([float((gf - m[f"{s}_gain_vs_forced"].reindex(c.index)).abs().max()),
                         float((gr - m[f"{s}_gain_vs_requested"].reindex(c.index)).abs().max())])
            named[f"{t}/{s}"] = {"max_abs_diff_gain_vs_forced": vals[-1][0], "max_abs_diff_gain_vs_requested": vals[-1][1]}
    r = pd.read_csv(root / "results/hotpot/tables/per_model_cot_req.csv").set_index("model_key")
    fo = pd.read_csv(root / "results/hotpot/tables/per_model_cot_forced.csv").set_index("model_key")
    mech = json.loads((root / "manifests/predictions_hotpot_mechanism.json").read_text(encoding="utf-8"))["per_model"]
    vals.append([float((fo["gain"] - pd.Series({k: v["gain_vs_forced"] for k, v in mech.items()}).reindex(fo.index))
                       .abs().max()),
                 float((r["intervention"].reindex(fo.index) - fo["intervention"]).abs().max())])
    named["hotpot/strict"] = {"max_abs_diff_gain_vs_forced": vals[-1][0], "max_abs_diff_cot_between_files": vals[-1][1]}
    return {"n": len(vals), "max": max(v for x in vals for v in x), "named": named}


class Facts:
    pass


def printed_lists(text: str) -> dict:
    """The quantities ladder.md prints, read from its own tables, never from the analysis's code: `placed`, every
    (benchmark, score, quantity) printed with a placement against the panel (section 6's rows, the readable rows as
    score "readable", and the glance tables' placed columns), and `family`, every (benchmark, score, quantity) row of the
    family tables (section 5). The cut-against-uncut checks run over these lists, so a quantity that is printed with a
    placement is compared between the readings whatever list the analysis keeps."""
    md = sections(text)
    sec = {k.split(".")[0]: v for k, v in md.items() if k[:1].isdigit()}
    placed = set()
    for _, h3, _, head, rows in tables_with_context(sec.get("6", [])):
        m = re.match(TASK_RE + r"(?:, (strict|content))? \((discovery draws|pooled rows)\)$", h3)
        if not m or head[:2] != ["Score", "Quantity"]:
            continue
        tk = TASK_ROW[m.group(1)]
        for row in rows:
            a = re.match(r"^Readable, (requested|forced|cot) \(%\)$", row[1]) if row[0] == "–" else None
            if a:
                placed.add((tk, "readable", a.group(1)))
            elif row[0] in SCORE_COL and row[1] in Q_OF:
                placed.add((tk, row[0], Q_OF[row[1]]))
    for _, _, _, head, rows in tables_with_context(sec.get("1", [])):
        if not head or head[0] != "Model":
            continue
        hot = head[1] != "Benchmark"
        tasks = ["hotpot"] if hot else sorted({TASK_ROW[r[1]] for r in rows if r[1] in TASK_ROW})
        for h in head:
            base = re.sub(r" \((discovery draws|pooled rows)\)$", "", h)
            if base in GLANCE_COLS:
                placed |= {(tk,) + GLANCE_COLS[base] for tk in tasks}
            elif base == "Requested readable (%)" and not hot:          # HotpotQA's is printed without a placement
                placed |= {(tk, "readable", "requested") for tk in tasks}
    family = set()
    for _, _, label, _, rows in tables_with_context(sec.get("5", [])):
        m = re.match(TASK_RE + r"(?: \(strict / content; discovery draws\)|, (strict|content) \((discovery draws|"
                     r"pooled rows)\)):$", label)
        if not m:
            continue
        tk = TASK_ROW[m.group(1)]
        scores = ("strict", "content") if m.group(2) is None else (m.group(2),)
        for row in rows:
            q = Q_OF.get(row[0])
            if q is not None:
                family |= {(tk, s, q) for s in (("strict",) if q == "w" else scores)}
    order = {t: i for i, t in enumerate(TASKS)}
    key = (lambda x: (order[x[0]], x[1], x[2]))
    return {"placed": sorted(placed, key=key), "family": sorted(family, key=key)}


def disc_value(x: dict, s: str, q: str):
    """A placed quantity of one reading on the discovery draws: a score's quantity, or the readable share of arm q."""
    if s == "readable":
        return g(x, "readable", q, "discovery")
    return value(x, s, "discovery", q)


def stop_ranges(F: Facts, keys: list[str], vals: list, t: str, s: str, q: str, reading: str):
    """Per rung, the (low, high) the stop's unread replies allow (the checker's own count from the raw replies,
    F.stop); None where the bound does not apply or no count is here."""
    if s != "content" or t not in STOP_TASKS or q not in STOP_Q or not F.stop:
        return None
    out = []
    for key, v in zip(keys, vals):
        src = "panel" if key not in FILES else "ladder"
        u = (F.stop.get(key) or {}).get(t, {}).get("discovery_pp")
        out.append(bound(v, u, q, rung_value(F, key, src, t, "strict", "w", reading)))
    return out


def build_facts(root: Path, J: dict, rec: dict, panel: dict, printed: dict | None = None,
                stop: dict | None = None) -> Facts:
    F = Facts()
    F.root, F.J, F.panel = root, J, panel
    F.stop = stop or {}
    F.printed = printed or {"placed": [], "family": []}
    F.x = rec                                                       # key -> reading -> task -> recomputed
    F.present = [k for k in FILES if any(F.x.get(k, {}).get(r) for r in READINGS)]
    F.missing = [k for k in FILES if k not in F.present]
    F.partial = [k for k in F.present if not all(t in F.x[k][r] for t in TASKS for r in READINGS)]
    disc, val = sampler_draws("gsm8k", root)
    F.n_disc, F.n_all = len(disc), len(disc) + len(val)
    sh = json.loads((root / "manifests/sampler_hotpot.json").read_text(encoding="utf-8"))
    F.hotpot_configs = len({str(c) for c in sh["discovery_config_ids"] + sh["validation_config_ids"]})
    F.comp = {k: completeness(root, k) for k in FILES}
    F.loop = {k: (None if k in API else loop_state(root, k)) for k in FILES}
    # placements, both readings, on the panel's basis
    F.place = {}
    for k in F.present:
        for r in READINGS:
            for t, x in F.x[k][r].items():
                for s in SCORE_COL:
                    for q in QUANTS:
                        if q == "w" and s == "content":
                            continue
                        col = "w" if q == "w" else f"{q}_{s}"
                        F.place[(k, t, r, s, q)] = place(panel_basis_value(x, t, s, q), panel[t][col])
                if t in DISC_TASKS:
                    for a in READ_ARMS:
                        F.place[(k, t, r, "readable", a)] = place(g(x, "readable", a, "discovery"),
                                                                  panel[t][f"readable_{a}"])
    # family ladders on the panel's basis, cut reading
    F.fam = {}
    for name, spec in FAMILIES.items():
        rungs = []
        for key, src in spec["rungs"]:
            present = src == "panel" or key in F.present
            rungs.append({"key": key, "src": src, "present": present})
        f = {"rungs": rungs, "present": [r for r in rungs if r["present"]], "vals": {}, "dir": {}}
        for t in TASKS:
            for s in SCORE_COL:
                for q in QUANTS:
                    if q == "w" and s == "content":
                        continue
                    vals = [rung_value(F, r["key"], r["src"], t, s, q, "cut") for r in f["present"]]
                    f["vals"][(t, s, q)] = vals
                    kv = [(r["key"], v) for r, v in zip(f["present"], vals) if v is not None]
                    clean = [v for _, v in kv]
                    rg = stop_ranges(F, [k for k, _ in kv], clean, t, s, q, "cut")
                    f.setdefault("ranges", {})[(t, s, q)] = (rg, [k for k, _ in kv], clean)
                    f["dir"][(t, s, q)] = direction(clean, "w" if q == "w" else "pp", rg)
        F.fam[name] = f
    # cut against uncut: every quantity ladder.md prints with a placement (its own tables' list), on the discovery
    # draws
    diffs = []
    for k in F.present:
        for t in TASKS:
            if t not in F.x[k]["cut"] or t not in F.x[k]["uncut"]:
                continue
            for tk, s, q in F.printed["placed"]:
                if tk != t:
                    continue
                a = disc_value(F.x[k]["cut"][t], s, q)
                b = disc_value(F.x[k]["uncut"][t], s, q)
                diffs.append({"model": k, "task": t, "score": s, "q": q, "cut": a, "uncut": b, "d": sub(b, a)})
    F.diffs = diffs
    # Pairs of ladder models, per (benchmark, score, quantity): a reversal when both readings order the pair, in
    # opposite ways; a tie that separates when one reading ties the pair exactly and the other orders it. A tie is
    # never counted as a reversal.
    groups = {}
    for d in diffs:
        if d["cut"] is not None and d["uncut"] is not None:
            groups.setdefault((d["task"], d["score"], d["q"]), []).append(d)
    F.flip_list, F.tie_list = pair_orders(groups)
    F.flips = len(F.flip_list)
    F.rank_changes = {"discovery": 0, "pooled": 0}
    F.rank_list = []
    for k in F.present:
        for t, s, q in F.printed["placed"]:
            a, b = F.place.get((k, t, "cut", s, q)), F.place.get((k, t, "uncut", s, q))
            if a and b and (a["below"], a["above"]) != (b["below"], b["above"]):
                F.rank_changes["discovery" if s == "readable" else basis_of(t, s)] += 1
                F.rank_list.append((k, t, s, q, a, b))
    F.fam_changes = []
    for name, spec in FAMILIES.items():
        for t, s, q in F.printed["family"]:
            cv, uv, any_l, ks = [], [], False, []
            for key, src in spec["rungs"]:
                if src == "panel":
                    if key not in panel[t].index:
                        continue
                    v = rung_value(F, key, src, t, s, q, "cut")
                    cv.append(v)
                    uv.append(v)
                    ks.append(key)
                    continue
                if key not in F.present or t not in F.x[key]["cut"] or t not in F.x[key]["uncut"]:
                    continue
                any_l = True
                cv.append(rung_value(F, key, src, t, s, q, "cut"))
                uv.append(rung_value(F, key, src, t, s, q, "uncut"))
                ks.append(key)
            if not any_l or None in cv or None in uv:
                continue
            kind = "w" if q == "w" else "pp"
            dc = direction(cv, kind, stop_ranges(F, ks, cv, t, s, q, "cut"))
            du = direction(uv, kind, stop_ranges(F, ks, uv, t, s, q, "uncut"))
            if dc != du:
                F.fam_changes.append({"family": name, "task": t, "score": s, "q": q, "basis": basis_of(t, s),
                                      "cut": cv, "uncut": uv, "dc": dc, "du": du})
    # BIG-Bench Hard forced gaps
    pg = (panel["bbh"]["forced_content"] - panel["bbh"]["forced_strict"]).dropna()
    F.bbh_panel_gap = pg
    F.bbh_gap = {k: place(F.x[k]["cut"]["bbh"]["gap_discovery"]["forced"], pg) for k in F.present
                 if "bbh" in F.x[k]["cut"]}
    F.fs_panel = fewshot_panel_gaps(root)
    F.fs_place, F.fs_unsettled = {}, {}
    if F.fs_panel:
        gaps = [x["gap"] for x in F.fs_panel["gaps"].values()]
        for k in F.bbh_gap:
            v = F.x[k]["cut"]["bbh"]["gap_all_rows"]["fs_forced"]
            F.fs_place[k] = place(v, gaps)
            F.fs_unsettled[k] = sorted(m for m, x in F.fs_panel["gaps"].items()
                                       if v is not None and abs(v - x["gap"]) <= Fraction(1, 100))
    F.other_max = {}
    for k in F.present:
        best = None
        for t in ("gsm8k", "math", "hotpot"):
            x = F.x[k]["cut"].get(t)
            if not x:
                continue
            for arm in ("forced", "fs_forced"):
                v = x["gap_all_rows"].get(arm)
                if v is not None and (best is None or abs(v) > abs(best[0])):
                    best = (v, t, arm)
        if best:
            F.other_max[k] = best
    F.sizes = {k: size_facts(root, k) for k in SIZE_SRC}
    F.spend = spend_facts(root)
    F.prec = panel_dtypes(root, list(panel["gsm8k"].index.union(panel["hotpot"].index)))
    F.kernels = {k: kernels(root, k) for k in FILES}
    F.equiv = [json.loads(p.read_text(encoding="utf-8")) | {"_file": p.relative_to(root).as_posix()}
               for p in sorted((root / "results" / "ladder").glob("equiv_*.json"))]
    F.panel_checks = panel_file_checks(root)
    F.shares = []
    for k in F.present:
        for t in TASKS:
            x = F.x[k]["cut"].get(t)
            if not x or not F.place.get((k, t, "cut", "content", "cot_minus_forced")):
                continue
            rd = g(x, "readable", "requested", "discovery")
            if rd is None or rd >= READABLE_BELOW:
                continue

            def share(sv, cv):
                return None if sv is None or cv is None or sv == 0 else 100 * (sv - cv) / sv
            gs, gc = (x[s]["discovery"]["cot_minus_requested"] for s in ("strict", "content"))
            rs, rc = (value(x, s, "discovery", "residual") for s in ("strict", "content"))
            F.shares.append({"key": k, "task": t, "readable": rd, "gap": x["gap_discovery"]["requested"],
                             "share_gain": share(gs, gc), "share_residual": share(rs, rc)})
    return F


def pair_orders(groups: dict) -> tuple[list, list]:
    """Per group of ladder models (one benchmark, score and quantity; each row with its cut and uncut value): the
    pairs that both readings order, in opposite ways (reversals), and the pairs that one reading ties exactly while the
    other orders them (ties that separate, with the reading that ties them). A tie is never a reversal."""
    flips, ties = [], []
    for rows in groups.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                c, u = sgn(a["cut"] - b["cut"]), sgn(a["uncut"] - b["uncut"])
                if c != 0 and u != 0 and c != u:
                    flips.append((a, b))
                elif (c == 0) != (u == 0):
                    ties.append((a, b, "cut" if c == 0 else "uncut"))
    return flips, ties


def largest(F: Facts, readable: bool | None = None) -> tuple:
    """The largest |uncut − cut| over the compared quantities and the set of every quantity that shares it (exact
    ratios compared exactly; a float within 1e-9); readable=False leaves out the readable shares."""
    ds = [d for d in F.diffs if d["d"] is not None and (readable is None or (d["score"] == "readable") == readable)]
    if not ds:
        return None, set()
    mx = max(abs(d["d"]) for d in ds)

    def eq(a, b):
        return a == b if isinstance(a, Fraction) and isinstance(b, Fraction) else abs(float(a) - float(b)) <= TOL
    return mx, {(d["model"], d["task"], d["score"], d["q"]) for d in ds if eq(abs(d["d"]), mx)}


def without_readable(scores: list) -> str:
    """' (n without the readable shares)' when some of the listed changes are readable shares."""
    n = sum(1 for s in scores if s != "readable")
    return f" ({n} without the readable shares)" if n != len(scores) else ""


def rung_value(F: Facts, key: str, src: str, t: str, s: str, q: str, reading: str):
    if src == "panel":
        pt = F.panel[t]
        if key not in pt.index:
            return None
        v = pt.at[key, "w" if q == "w" else f"{q}_{s}"]
        return None if (isinstance(v, float) and math.isnan(v)) else v
    x = F.x.get(key, {}).get(reading, {}).get(t)
    return None if x is None else panel_basis_value(x, t, s, q)


def check_json_extras(T: Tally, F: Facts) -> None:
    """Part 1 continued: the family tables, placements (both readings), BIG-Bench Hard gaps, cut against uncut,
    completeness and the DeepSeek reproduction count against ladder.json."""
    J = F.J
    for what, (theirs, mine) in ((F.fs_panel or {}).get("scorer_own") or {}).items():
        T.text(f"{FEWSHOT_PANEL}: Mistral Large 3 five-shot forced, {what}, the file's scoring against the saved rows",
               mine, theirs)
    for fam, f in J.get("families", {}).items():
        for row in f.get("rungs", []) + f.get("context", []):
            k, src = row["key"], row["source"]
            for t in TASKS:
                for s in SCORE_COL:
                    for q in QUANTS:
                        if q == "w" and s == "content":
                            continue
                        mine = rung_value(F, k, src, t, s, q, "cut") if (src == "panel" or k in F.present) else None
                        T.num(f"family/{fam}/{k}/{t}/{s}/{q}", mine, g(row, "values", t, s, q))
        for t, s, q in F.printed["family"]:                       # the rows section 5 prints
            jd = g(f, "direction", t, s, q)
            T.text(f"family/{fam}/{t}/{s}/{q}/direction", F.fam[fam]["dir"][(t, s, q)], g(jd, "direction"))
    for (k, t, r, s, q), p in F.place.items():
        jp = g(J, "placement", k, t, r, s, q) if s != "readable" else g(J, "placement", k, t, r, "readable", q)
        base = f"placement/{k}/{t}/{r}/{s}/{q}"
        T.num(f"{base}/value", p["value"], g(jp, "value"))
        T.num(f"{base}/panel_n", p["n"], g(jp, "panel_n"))
        T.num(f"{base}/panel_median", p["median"], g(jp, "panel_median"))
        T.num(f"{base}/below", p["below"], g(jp, "below"))
        T.num(f"{base}/above", p["above"], g(jp, "above"))
        T.text(f"{base}/outside", {" ↑": "above", " ↓": "below", "": None}[p["mark"]], g(jp, "outside_panel_range"))
    for k, p in F.bbh_gap.items():
        jp = g(J, "bbh_forced_gap", "models", k, "zero_shot_forced_discovery")
        T.num(f"bbh_forced_gap/{k}/value", p["value"], g(jp, "value"))
        T.num(f"bbh_forced_gap/{k}/below", p["below"], g(jp, "below"))
        T.num(f"bbh_forced_gap/{k}/fewshot_all_rows", F.x[k]["cut"]["bbh"]["gap_all_rows"]["fs_forced"],
              g(J, "bbh_forced_gap", "models", k, "fewshot_forced_all_rows"))
    for k, p in F.fs_place.items():
        jp = g(J, "bbh_forced_gap", "models", k, "fewshot_forced_all_rows_placement")
        for f_, j_ in (("value", "value"), ("below", "below"), ("above", "above"), ("n", "panel_n"),
                       ("median", "panel_median"), ("min", "panel_min"), ("max", "panel_max")):
            T.num(f"bbh_forced_gap/{k}/fewshot placement/{f_}", p[f_], g(jp, j_))
        T.text(f"bbh_forced_gap/{k}/fewshot placement/outside", {" ↑": "above", " ↓": "below", "": None}[p["mark"]],
               g(jp, "outside_panel_range"))
        T.text(f"bbh_forced_gap/{k}/fewshot within the panel's rounding", F.fs_unsettled[k],
               g(J, "bbh_forced_gap", "models", k, "fewshot_within_panel_rounding"))
    for k, (v, t, arm) in F.other_max.items():
        T.num(f"forced_other_max/{k}", v, g(J, "forced_other_max", k, "gap_pp"))
        T.text(f"forced_other_max/{k}/where", f"{t}/{arm}",
               f"{g(J, 'forced_other_max', k, 'task')}/{g(J, 'forced_other_max', k, 'arm')}")
    cv = J.get("cut_vs_uncut", {})
    ds = [d for d in F.diffs if d["d"] is not None]
    T.num("cut_vs_uncut/max_abs_difference", max((abs(d["d"]) for d in ds), default=None),
          cv.get("max_abs_difference"))
    T.num("cut_vs_uncut/differences", len(F.diffs), len(cv.get("differences", [])))
    T.text("cut_vs_uncut/differences (as a set)",
           sorted((d["model"], d["task"], d["score"], d["q"]) for d in F.diffs),
           sorted((d.get("model"), d.get("task"), d.get("score"), d.get("quantity")) for d in cv.get("differences", [])))
    for rd, mk, lk in ((None, "max_abs_difference", "largest"), (False, "max_abs_difference_scores", "largest_scores")):
        mx, top = largest(F, rd)
        if rd is not None:
            T.num(f"cut_vs_uncut/{mk}", mx, cv.get(mk))
        T.text(f"cut_vs_uncut/{lk}", sorted(top), sorted((x.get("model"), x.get("task"), x.get("score"),
                                                          x.get("quantity")) for x in cv.get(lk, [])))
    T.num("cut_vs_uncut/ordering_flips", F.flips, len(cv.get("ordering_flips", [])))
    for b in ("discovery", "pooled"):
        T.num(f"cut_vs_uncut/panel_rank_changes/{b}", F.rank_changes[b],
              sum(1 for x in cv.get("panel_rank_changes", []) if x.get("basis") == b))
        T.num(f"cut_vs_uncut/family_direction_changes/{b}", sum(1 for x in F.fam_changes if x["basis"] == b),
              sum(1 for x in cv.get("family_direction_changes", []) if x.get("basis") == b))
    for k in FILES:
        c = g(J, "completeness", k)
        T.num(f"completeness/{k}/design_rows", F.comp[k]["design_rows"], g(c, "design_rows"))
        T.text(f"completeness/{k}/exact", F.comp[k]["exact"], g(c, "exact"))
        for r in READINGS:
            T.num(f"completeness/{k}/scored/{r}", F.comp[k]["scored"][r], g(c, "scored_rows", r))


def reproduction_count(F: Facts) -> tuple[int, int] | None:
    """The DeepSeek block of ladder.json against results/frontier_panel/nebius_panel.json, leaf by leaf (tolerance
    1e-9): (values compared, values that differ)."""
    p = F.root / NEBIUS_JSON
    mj = g(F.J, "models", DS)
    if not p.exists() or mj is None or mj.get("status") == "missing":
        return None
    neb = json.loads(p.read_text(encoding="utf-8"))
    n, bad = 0, 0

    def cmp(a, b):
        nonlocal n, bad
        n += 1
        if a is None or b is None:
            bad += (a is None) != (b is None)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            bad += abs(float(a) - float(b)) > TOL
        else:
            bad += a != b
    for t in TASKS:
        for r, src in (("cut", "tasks"), ("uncut", "tasks_uncut")):
            rep, mine = g(neb, src, t), g(mj, "tasks", t, r)
            if rep is None or mine is None:
                bad += 1
                continue
            for k in ("configs_design", "configs_answered", "units_answered"):
                cmp(mine.get(k), rep.get(k))
            for k in ("discovery", "all_draws"):
                cmp(g(mine, "w", k), g(rep, "w", k))
            for s in SCORE_COL:
                for lab in ("all_draws", "discovery"):
                    for a in ("requested", "forced", "cot"):
                        cmp(g(mine, s, lab, "accuracy_pp", a), g(rep, s, lab, "accuracy_pp", a))
                    for k in ("gain_over_requested_pp", "gain_over_forced_pp", "corrected_gain_pp"):
                        cmp(g(mine, s, lab, k), g(rep, s, lab, k))
                for q, v in (g(rep, s, "template") or {}).items():
                    for k, x in v.items():
                        cmp(g(mine, s, "template", q, k), x)
            for arm, d in (g(neb, "fewshot", t, r) or {}).items():
                for k, x in d.items():
                    cmp(g(mine, "fewshot", arm, k), x)
    return n, bad


# ----------------------------------------------------------------------------------------------------------------
# DeepSeek against results/frontier_panel/nebius_panel.md
# ----------------------------------------------------------------------------------------------------------------

def md_tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """(heading or the paragraph before it, header cells, rows) of every Markdown table in the text."""
    out, lines, i, ctx = [], text.splitlines(), 0, ""
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-:| ]+\|$", lines[i + 1]):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append((ctx, head, rows))
            continue
        if ln.strip():
            ctx = ln.strip()
        i += 1
    return out


def fmt(v, d: int) -> str:
    if v is None:
        return "–"
    return f"{v:.{d}f}"


def check_nebius_md(T: Tally, x: dict, md_path: Path) -> int:
    """x: recomputed DeepSeek values per reading and benchmark. Compares every cell of the md's main tables (cut and
    uncut) and of its five-shot table, at the digits printed. Returns the number of cells compared."""
    n0 = T.n
    tabs = md_tables(md_path.read_text(encoding="utf-8"))
    main = [(c, h, r) for c, h, r in tabs if h[:3] == ["Benchmark", "Score", "Requested"]]
    if len(main) < 2:
        T.n += 1
        T.bad.append(f"{md_path}: the two main tables (cut, uncut) were not found")
        return T.n - n0
    for reading, (_, head, rows) in zip(READINGS, main[:2]):
        for row in rows:
            t, s = TASK_ROW[row[0]], row[1]
            r = x[reading][t]
            a, d = r[s]["all_draws"], r[s]["discovery"]
            mine = {"Requested": fmt(a["acc"]["requested"], 1), "Forced": fmt(a["acc"]["forced"], 1),
                    "CoT": fmt(a["acc"]["cot"], 1), "CoT − requested": fmt(a["cot_minus_requested"], 1),
                    "CoT − forced": fmt(a["cot_minus_forced"], 1), "w (disc.)": fmt(r["w"]["discovery"], 3),
                    "Corrected gain (disc.)": fmt(d["corrected_gain"], 1),
                    "CoT − requested (disc.)": fmt(d["cot_minus_requested"], 1)}
            for col, cell in zip(head[2:], row[2:]):
                T.text(f"nebius_panel.md/{reading}/{t}/{s}/{col}", mine[col], cell)
    fs = [(c, h, r) for c, h, r in tabs if h[:3] == ["Benchmark", "Arm", "Reading"]]
    if not fs:
        T.n += 1
        T.bad.append(f"{md_path}: the five-shot table was not found")
        return T.n - n0
    _, head, rows = fs[0]
    for row in rows:
        t, arm, reading = TASK_ROW[row[0]], row[1], row[2]
        d = x[reading][t]["fewshot"].get(arm, {})
        mine = {"Rows": f"{d.get('rows', 0):,}", "C": str(d.get("configs")),
                "Strict": fmt(d.get("strict_pp_config_mean"), 1), "Content": fmt(d.get("content_pp_config_mean"), 1),
                "Strict (draws)": fmt(d.get("strict_pp_all_draws"), 1),
                "Content (draws)": fmt(d.get("content_pp_all_draws"), 1),
                "Share longer than forced cap": fmt(None if d.get("share_longer_than_forced_cap") is None
                                                    else 100 * d["share_longer_than_forced_cap"], 1)}
        for col, cell in zip(head[3:], row[3:]):
            T.text(f"nebius_panel.md/fewshot/{reading}/{t}/{arm}/{col}", mine[col], cell)
    return T.n - n0


# ----------------------------------------------------------------------------------------------------------------
# Part 2: what is printed
# ----------------------------------------------------------------------------------------------------------------

NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")
# A parameter count glued to its unit (8.92B, 675B) after a space or bracket: a number too, so it cannot slip past.
SIZE_NUM_RE = re.compile(r"(?<![\w.\-])\d+(?:\.\d+)?B\b")
V = r"(-?\d+(?:\.\d+)?)"
VI = r"(\d+(?:,\d{3})*)"
CELL_RE = re.compile(r"^(-?\d+\.\d+) \((\d+)/(\d+)(?:, (\d+) tied)?\)( ↑| ↓)?$")
CELL_PAT = r"-?\d+\.\d+ \(\d+/\d+(?:, \d+ tied)?\)(?: ↑| ↓)?"
VALS = r"(-?\d+\.\d(?: → -?\d+\.\d)*)"
STEPW = r"(?:up|down|flat)(?: within the stop's loss)?"
DIRW = r"(rises|falls|flat|" + STEPW + r"(?:, " + STEPW + r")+|(?:up|down|flat) within the stop's loss|–)"
STAMP = r"(\d{4}-\d\d-\d\d \d\d:\d\d(?::\d\d)?)"


def alt(words) -> str:
    return "(" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + ")"


NAME_RE = alt(KEY_OF)
NAME_NC = "(?:" + NAME_RE[1:]
TASK_RE = alt(TASK_ROW)
FAM_RE = alt(FAMILIES)
QLAB_RE = alt(Q_OF)
# Structural spans: labels and references, not quantities.
MASK_RES = [re.compile(p) for p in (
    r"`[^`]*`", NAME_RE, TASK_RE, FAM_RE, alt(SIZE.values()), alt(PANEL_RUNGS),
    r"sections? \d+(?:(?:, | and )\d+)*", r"footnotes? F\d(?:, F\d)*", r"\bF\d\b", r"\bitem \d+",
    r"\(1 − (?:discovery )?w(?: over the 48 draws)?\)", r"D\d+, note of \d+ November", r"\b16-bit\b",
    r"with w near 0", r"\bMinistral 3\b", r"Llama-3\.1-70B", r"DeepSeek V4\.1-Flash", r"\bLarge 3\b",
    r"\bitems? \d+(?:–\d+)?", r"Corrections \d+–\d+", r"\[0, 1\]", r"C − 1", r"RTX PRO 6000",
    r"Qwen3-2507|Qwen3-235B|gpt-oss-120b", r"Mistral-Large-3-675B-Instruct-2512", r"[\w./-]*V4\.1-Flash-1017[\w./-]*",
    r"\brevision [0-9a-f]{40}\b", alt(FILES))]


class Line:
    def __init__(self, text: str, where: str):
        self.text, self.where = text, where
        self.used = [False] * len(text)

    def use(self, a: int, b: int) -> None:
        for i in range(a, b):
            self.used[i] = True

    def mask_structural(self) -> None:
        for rx in MASK_RES:
            for m in rx.finditer(self.text):
                self.use(*m.span())

    def leftovers(self) -> list[str]:
        """Numbers on the line that no handler checked and no structural label covers."""
        out = [m.group(0) for m in NUM_RE.finditer(self.text) if not all(self.used[m.start():m.end()])]
        return out + [m.group(0) for m in SIZE_NUM_RE.finditer(self.text) if not all(self.used[m.start():m.end() - 1])]


class _Shifted:
    """A Line seen through a shifted text: spans of the shifted text are marked at their place in the line."""
    def __init__(self, ln: Line, shift: int):
        self._ln, self._shift, self.where, self.text = ln, shift, ln.where, ln.text[shift:]

    def use(self, a: int, b: int) -> None:
        self._ln.use(a + self._shift, b + self._shift)


class TextCheck:
    def __init__(self, T: Tally, F: Facts):
        self.T, self.F = T, F
        self.s9 = {"flips": [], "ties": [], "ranks": [], "fams": []}

    # -- helpers --------------------------------------------------------------------------------------------------
    def p(self, ln: Line, m, grp, v, what: str, off: int = 0) -> None:
        """A printed number (regex group) against a value, at the printed precision; marks it checked."""
        a, b = m.span(grp)
        ln.use(a + off, b + off)
        self.T.printed(f"{ln.where}: {what}", m.group(grp), v)

    def w(self, ln: Line, m, grp, want, what: str, off: int = 0) -> None:
        """A printed word or phrase (regex group) against the expected text; marks it checked."""
        a, b = m.span(grp)
        ln.use(a + off, b + off)
        self.T.text(f"{ln.where}: {what}", want, m.group(grp))

    def fail(self, ln: Line, why: str) -> None:
        self.T.n += 1
        self.T.bad.append(f"{ln.where}: {why}: {ln.text[:160]}")

    def finish(self, ln: Line) -> None:
        for tok in ln.leftovers():
            self.T.n += 1
            self.T.bad.append(f"{ln.where}: number not checked: {tok!r} in {ln.text[:120]!r}")

    def cell(self, where: str, printed: str, pl: dict, d: int | None = None) -> None:
        """A placement cell: value (b/M) and its mark."""
        if pl is None or pl["value"] is None:
            self.T.text(f"{where}: cell", "–", printed)
            return
        m = CELL_RE.match(printed)
        if not m:
            self.T.n += 1
            self.T.bad.append(f"{where}: cell not of the form value (b/M) mark: {printed!r}")
            return
        self.T.printed(f"{where}: value", m.group(1), pl["value"])
        self.T.printed(f"{where}: below", m.group(2), pl["below"])
        self.T.printed(f"{where}: panel n", m.group(3), pl["n"])
        self.T.printed(f"{where}: tied", m.group(4) or "0", pl["equal"])           # ties are printed, never hidden
        self.T.text(f"{where}: mark", pl["mark"], m.group(5) or "")

    def vals(self, ln: Line, m, grp, want: list, what: str, off: int = 0) -> None:
        a, _ = m.span(grp)
        parts = m.group(grp).split(" → ")
        if len(parts) != len(want):
            self.fail(ln, f"{what}: {len(parts)} values printed, {len(want)} recomputed")
            ln.use(a + off, a + off + len(m.group(grp)))
            return
        pos = a + off
        for i, (s, v) in enumerate(zip(parts, want)):
            self.T.printed(f"{ln.where}: {what}[{i}]", s, v)
            pos = ln.text.index(s, pos)
            ln.use(pos, pos + len(s))
            pos += len(s)

    def state(self, ln: Line, key: str, text: str, off: int) -> None:
        """A model's completeness state as printed, against the request files, the scored rows and the loop log."""
        c, lp = self.F.comp[key], self.F.loop.get(key)
        tot = c["scored"]
        if c["exact"]:
            kind = "exact"
        elif tot["cut"] == 0 and tot["uncut"] == 0:
            kind = "none"
        else:
            kind = "short"
        found = None
        m = re.search(r"every design row scored once in both readings \(" + VI + r"\)", text)
        if m:
            found = "exact"
            self.p(ln, m, 1, c["design_rows"], f"{key} design rows", off)
        m = re.search(r"no scored rows here \(" + VI + " of " + VI + r"\)", text)
        if m:
            found = "none"
            self.p(ln, m, 1, tot["cut"] + tot["uncut"], f"{key} scored rows", off)
            self.p(ln, m, 2, c["design_rows"], f"{key} design rows", off)
        m = re.search(r"short: cut " + VI + " and uncut " + VI + " of " + VI + " rows", text)
        if m:
            found = "short"
            self.p(ln, m, 1, tot["cut"], f"{key} scored cut", off)
            self.p(ln, m, 2, tot["uncut"], f"{key} scored uncut", off)
            self.p(ln, m, 3, c["design_rows"], f"{key} design rows", off)
        m = re.search(r"cluster jobs done (\d+)/(\d+) at the loop's pass of " + STAMP, text)
        if m:
            jd = (lp or {}).get("jobs_done") or "?/?"
            self.w(ln, m, 1, jd.split("/")[0], f"{key} jobs done", off)
            self.w(ln, m, 2, jd.split("/")[-1], f"{key} jobs total", off)
            self.w(ln, m, 3, (lp or {}).get("pass"), f"{key} loop pass", off)
        elif kind == "none" and lp and lp.get("jobs_done"):
            self.fail(ln, f"{key}: the loop's job count is not printed")
        self.T.text(f"{ln.where}: {key} state kind", kind, found)

    def names_list(self, ln: Line, m, grp, keys: list[str], what: str, short: bool = False) -> None:
        want = ", ".join(NAMES[k][0 if short else 1] for k in keys)
        self.w(ln, m, grp, want, what)

    # -- section 1 findings and the note's copies ----------------------------------------------------------------
    def finding(self, ln: Line) -> None:
        t, F = ln.text, self.F
        if (m := re.match(r"^(\d+) of (\d+) ladder models are in(?:: (" + NAME_NC + "(?:, " + NAME_NC + r")*))?\.", t)):
            self.p(ln, m, 1, len(F.present), "models in")
            self.p(ln, m, 2, len(FILES), "ladder models")
            if m.group(3) is not None:
                self.names_list(ln, m, 3, F.present, "models in")
            rest_off = m.end()
            pm = re.search(r" Partial: ([^.]*)\.", t[rest_off:])
            if pm:
                self.w(ln, pm, 1, ", ".join(NAMES[k][1] for k in F.partial), "partial", rest_off)
            spans = [(mm.start(), mm.end(), KEY_OF[mm.group(1)])
                     for mm in re.finditer(NAME_RE + ": ", t[rest_off:])]
            got = []
            for i, (a, b, key) in enumerate(spans):
                end = spans[i + 1][0] if i + 1 < len(spans) else len(t) - rest_off
                self.state(ln, key, t[rest_off + b: rest_off + end], rest_off + b)
                got.append(key)
            self.T.text(f"{ln.where}: missing models listed", F.missing, got)
            return
        if (m := re.match(r"^Completeness: (\d+) of (\d+) models in have every one of the " + VI + " design rows", t)):
            self.p(ln, m, 1, sum(F.comp[k]["exact"] for k in F.present), "exact models")
            self.p(ln, m, 2, len(F.present), "models in")
            self.p(ln, m, 3, max(F.comp[k]["design_rows"] for k in FILES), "design rows")
            for mm in re.finditer(NAME_RE + r" \((short: [^)]*)\)", t):
                self.state(ln, KEY_OF[mm.group(1)], mm.group(2), mm.start(2))
            return
        if (m := re.match(FAM_RE + r" ladder", t)):
            self.family_line(ln, m.group(1))
            return
        if t.startswith("Family steps within the stop's loss"):
            self.loss_line(ln)
            return
        if t.startswith("The stop's loss is not checked"):
            self.T.text(f"{ln.where}: the stop's loss is not checked here either", True, not F.stop)
            return
        if " against the panel" in t and (m := re.match(NAME_RE + " against the panel", t)):
            self.against(ln, KEY_OF[m.group(1)])
            return
        if t.startswith("Baseline shift"):
            self.residual(ln)
            return
        if t.startswith("Requested replies the strict rule cannot read"):
            self.readable(ln)
            return
        if t.startswith("BIG-Bench Hard zero-shot forced arm"):
            self.bbh0(ln)
            return
        if t.startswith("BIG-Bench Hard five-shot forced arm"):
            self.bbh5(ln)
            return
        if t.startswith("Cut against uncut"):
            self.cut_uncut(ln)
            return
        if t.startswith("DeepSeek V4.1-Flash-1017 is a single model"):
            self.ds_size(ln)
            return
        if t.startswith("Placements against the panel that move"):
            m = re.match(r"^Placements against the panel that move when the uncut values are used: (\d+) on the "
                         r"discovery draws( \((\d+) without the readable shares\))?, (\d+) on pooled rows \(HotpotQA "
                         r"content\)\. Family directions \(section 5\) that change: (\d+) on the discovery draws, (\d+) on "
                         r"pooled rows \(section 9\)\.$", t)
            if not m:
                return self.fail(ln, "placement-moves sentence not parsed")
            self.p(ln, m, 1, F.rank_changes["discovery"], "placements moved, discovery")
            disc = [x[2] for x in F.rank_list if x[2] == "readable" or basis_of(x[1], x[2]) == "discovery"]
            self.T.text(f"{ln.where}: count without the readable shares printed", without_readable(disc) != "",
                        m.group(2) is not None)
            if m.group(3) is not None:
                self.p(ln, m, 3, sum(1 for s in disc if s != "readable"), "placements moved, discovery, no readable")
            self.p(ln, m, 4, F.rank_changes["pooled"], "placements moved, pooled")
            self.p(ln, m, 5, sum(x["basis"] == "discovery" for x in F.fam_changes), "family changes, discovery")
            self.p(ln, m, 6, sum(x["basis"] == "pooled" for x in F.fam_changes), "family changes, pooled")
            return
        if t.startswith("Findings above that depend on the cap"):
            self.cap(ln)
            return
        if t.startswith("DeepSeek reproduction"):
            m = re.match(r"^DeepSeek reproduction: (\d+) values compared with `[^`]+`, (\d+) differ \(tolerance "
                         r"(1e-0?9)\)\.$", t)
            rc = reproduction_count(F)
            if not m or rc is None:
                return self.fail(ln, "reproduction sentence not parsed or no reproduction here")
            self.p(ln, m, 1, rc[0], "reproduction values compared")
            self.p(ln, m, 2, rc[1], "reproduction values differing")
            self.w(ln, m, 3, f"{TOL:g}", "reproduction tolerance")
            return

    def loss_steps(self) -> list[tuple]:
        """The family steps the stop's unread replies could turn (the checker's own counts and rule)."""
        out = []
        for fam, f in self.F.fam.items():
            for t in STOP_TASKS:
                for q in ("forced", "cot_minus_forced", "corrected_gain"):
                    rg, keys, vals = f.get("ranges", {}).get((t, "content", q), (None, [], []))
                    if not rg:
                        continue
                    for j in range(len(vals) - 1):
                        (alo, ahi), (blo, bhi) = rg[j], rg[j + 1]
                        if step_word(blo - ahi, "pp") != step_word(bhi - alo, "pp"):
                            out.append((fam, t, q, keys[j], keys[j + 1], vals[j + 1] - vals[j], blo - ahi, bhi - alo))
        return out

    def loss_line(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.match(r"^Family steps within the stop's loss \(content, discovery draws; the zero-shot forced replies "
                     r"the stop at the first `\}` left unread, counted none or all correct; footnote F1\)(?:: (none)\.|"
                     r": (.*)\. The stop cut these answers inside their box and the readers do not read them, so the "
                     r"forced arm does not settle the direction of (?:this step|these steps)\.)$", t)
        if not m:
            return self.fail(ln, "stop-loss line not parsed")
        want = self.loss_steps()
        if m.group(1):
            self.T.text(f"{ln.where}: family steps within the stop's loss", [], want)
            return
        qn = {"forced": "forced", "CoT − forced": "cot_minus_forced", "corrected gain": "corrected_gain"}
        items = list(re.finditer(FAM_RE + " " + TASK_RE + r" (forced|CoT − forced|corrected gain), (.+?) → (.+?): " + V
                                 + r" pp, between " + V + " and " + V + r" pp \(unread: " + VI + r" repl(?:y|ies), " + V
                                 + r" pp, of the (.+?)'s arm; " + VI + r" repl(?:y|ies), " + V + r" pp, of the (.+?)'s\)",
                                 m.group(2)))
        self.T.text(f"{ln.where}: family steps within the stop's loss",
                    [(x[0], x[1], x[2]) for x in want],
                    [(i.group(1), TASK_ROW[i.group(2)], qn[i.group(3)]) for i in items])
        off = m.start(2)
        for i, x in zip(items, want):
            fam, tk, q, a, b, step, lo, hi = x
            what = f"stop-loss step {fam} {tk} {q}"
            self.w(ln, i, 4, SIZE[a], f"{what} from", off)
            self.w(ln, i, 5, SIZE[b], f"{what} to", off)
            self.p(ln, i, 6, step, f"{what} step", off)
            self.p(ln, i, 7, lo, f"{what} low", off)
            self.p(ln, i, 8, hi, f"{what} high", off)
            self.p(ln, i, 9, F.stop[a][tk]["unread"], f"{what} unread replies, {a}", off)
            self.p(ln, i, 10, F.stop[a][tk]["discovery_pp"], f"{what} unread pp, {a}", off)
            self.w(ln, i, 11, SIZE[a], f"{what} unread rung", off)
            self.p(ln, i, 12, F.stop[b][tk]["unread"], f"{what} unread replies, {b}", off)
            self.p(ln, i, 13, F.stop[b][tk]["discovery_pp"], f"{what} unread pp, {b}", off)
            self.w(ln, i, 14, SIZE[b], f"{what} unread rung", off)

    def family_line(self, ln: Line, fam: str) -> None:
        t, f = ln.text, self.F.fam[fam]
        present = f["present"]
        miss = [r["key"] for r in f["rungs"] if not r["present"]]
        m = re.match(re.escape(fam) + r" ladder: (\d+) of (\d+) rungs in; no within-release direction can be read "
                     r"yet(?: \(waits for ([^)]*)\))?\.$", t)
        if m:
            self.p(ln, m, 1, len(present), f"{fam} rungs in")
            self.p(ln, m, 2, len(f["rungs"]), f"{fam} rungs")
            if m.group(3) is not None:
                self.w(ln, m, 3, ", ".join(NAMES[k][0] for k in miss), f"{fam} waits for")
            self.T.text(f"{ln.where}: {fam} fewer than two rungs", True, len(present) < 2)
            return
        m = re.match(re.escape(fam) + r" ladder \((.+?); (\d+) of (\d+) rungs\), content, discovery draws: (.*)\. "
                     r"HotpotQA, content, pooled rows: (.*?)\.(?: Waits for (.*)\.)?$", t)
        if not m:
            return self.fail(ln, f"{fam} family line not parsed")
        self.w(ln, m, 1, " → ".join(SIZE[r["key"]] for r in present), f"{fam} rung labels")
        self.p(ln, m, 2, len(present), f"{fam} rungs in")
        self.p(ln, m, 3, len(f["rungs"]), f"{fam} rungs")
        if m.group(6) is not None:
            self.w(ln, m, 6, ", ".join(NAMES[k][0] for k in miss), f"{fam} waits for")
        parts = {}
        off = m.start(4)
        for pm in re.finditer(TASK_RE + r": (forced .*?)(?=; (?:GSM8K|MATH-500|BIG-Bench Hard): |$)", m.group(4)):
            parts[TASK_ROW[pm.group(1)]] = (pm.group(2), off + pm.start(2))
        parts["hotpot"] = (m.group(5), m.start(5))
        self.T.text(f"{ln.where}: {fam} benchmarks", TASKS, [x for x in parts])
        for task, (txt, o) in parts.items():
            pm = re.match(r"^forced " + VALS + r" \(" + DIRW + r"\), CoT − forced " + VALS + r" \(" + DIRW
                          + r"\), CoT − requested " + VALS + r" \(" + DIRW + r"\)$", txt)
            if not pm:
                self.fail(ln, f"{fam} {task} part not parsed")
                continue
            for gi, q in ((1, "forced"), (3, "cot_minus_forced"), (5, "cot_minus_requested")):
                vv = [v for v in f["vals"][(task, "content", q)] if v is not None]
                self.vals(ln, pm, gi, vv, f"{fam} {task} content {q}", o)
                self.w(ln, pm, gi + 1, f["dir"][(task, "content", q)] or "–", f"{fam} {task} content {q} direction",
                       o)

    def tie(self, ln: Line, m, grp, p: dict, what: str, off: int = 0) -> None:
        """The printed tie count of a placement (group absent: none printed) against the recomputed one."""
        if m.group(grp) is None:
            self.T.printed(f"{ln.where}: {what} tied", "0", p["equal"])
        else:
            self.p(ln, m, grp, p["equal"], f"{what} tied", off)

    def against(self, ln: Line, key: str) -> None:
        t, F = ln.text, self.F
        pl = {tk: F.place.get((key, tk, "cut", "content", "cot_minus_forced")) for tk in TASKS}
        dk = [tk for tk in DISC_TASKS if pl[tk]]
        m = re.match(re.escape(NAMES[key][1]) + r" against the panel, CoT − forced \(content\), discovery draws: above "
                     r"the panel median on (\d+) of (\d+) benchmarks; (.*?)\.(?: HotpotQA, CoT − forced \(content\), "
                     r"pooled rows: " + V + r" pp, above (\d+) of (\d+)(?:, (\d+) tied)?, (above|not above) the panel "
                     r"median\.)?$", t)
        if not m:
            return self.fail(ln, f"{key} against-the-panel line not parsed")
        above = [tk for tk in dk if pl[tk]["value"] is not None and pl[tk]["value"] > pl[tk]["median"] + TOL]
        self.p(ln, m, 1, len(above), f"{key} above median")
        self.p(ln, m, 2, len(dk), f"{key} benchmarks")
        items = list(re.finditer(TASK_RE + " " + V + r" pp, above (\d+) of (\d+)(?:, (\d+) tied)?", m.group(3)))
        self.T.text(f"{ln.where}: {key} benchmarks listed", dk, [TASK_ROW[i.group(1)] for i in items])
        for i in items:
            p_ = pl[TASK_ROW[i.group(1)]]
            self.p(ln, i, 2, p_["value"], f"{key} {i.group(1)} CoT − forced content", m.start(3))
            self.p(ln, i, 3, p_["below"], f"{key} {i.group(1)} below", m.start(3))
            self.p(ln, i, 4, p_["n"], f"{key} {i.group(1)} panel n", m.start(3))
            self.tie(ln, i, 5, p_, f"{key} {i.group(1)}", m.start(3))
        if m.group(4) is not None:
            p_ = pl["hotpot"]
            self.p(ln, m, 4, p_["value"], f"{key} HotpotQA CoT − forced content (pooled)")
            self.p(ln, m, 5, p_["below"], f"{key} HotpotQA below")
            self.p(ln, m, 6, p_["n"], f"{key} HotpotQA panel n")
            self.tie(ln, m, 7, p_, f"{key} HotpotQA")
            self.w(ln, m, 8, "above" if p_["value"] is not None and p_["value"] > p_["median"] + TOL else "not above",
                   f"{key} HotpotQA median")
        elif pl["hotpot"]:
            self.fail(ln, f"{key}: HotpotQA placement not printed")

    def residual(self, ln: Line) -> None:
        t, F = ln.text, self.F
        rows = [(k, tk) for k in F.present for tk in TASKS if F.place.get((k, tk, "cut", "content", "residual"))]

        def within(tasks):
            return [(k, tk) for k, tk in rows if tk in tasks and F.place[(k, tk, "cut", "content", "residual")]["value"]
                    is not None and abs(F.place[(k, tk, "cut", "content", "residual")]["value"]) <= RESIDUAL_BAND]

        def names(cells):
            return ", ".join(f"{NAMES[k][1]} {TASK_NAME[tk]}" for k, tk in cells)
        m = re.search(r"Content, discovery draws \(GSM8K, MATH-500, BIG-Bench Hard\): within ±" + V + r" pp in (\d+) "
                      r"of (\d+) \(model, benchmark\) cells(?: \(([^)]*)\))?\.", t)
        if not m:
            return self.fail(ln, "residual sentence (discovery draws) not parsed")
        sd = within(DISC_TASKS)
        self.p(ln, m, 1, RESIDUAL_BAND, "residual band")
        self.p(ln, m, 2, len(sd), "residual within band, discovery")
        self.p(ln, m, 3, sum(tk in DISC_TASKS for _, tk in rows), "residual cells, discovery")
        if m.group(4) is not None:
            self.w(ln, m, 4, names(sd), "residual within band, discovery: cells")
        m = re.search(r"HotpotQA content, pooled rows \(the corrected gain there is \(1 − discovery w\) × pooled CoT − "
                      r"forced\): within ±" + V + r" pp in (\d+) of (\d+) cells(?: \(([^)]*)\))?\.", t)
        hr = [(k, tk) for k, tk in rows if tk == "hotpot"]
        if hr and not m:
            return self.fail(ln, "residual sentence (HotpotQA pooled) not parsed")
        if m:
            sh = within(["hotpot"])
            self.p(ln, m, 1, RESIDUAL_BAND, "residual band")
            self.p(ln, m, 2, len(sh), "residual within band, pooled")
            self.p(ln, m, 3, len(hr), "residual cells, pooled")
            if m.group(4) is not None:
                self.w(ln, m, 4, names(sh), "residual within band, pooled: cells")
        for label, want in (("discovery draws", [(k, tk, s) for k, tk in rows for s in SCORE_COL
                                                 if basis_of(tk, s) == "discovery"]),
                            ("HotpotQA content, pooled rows", [(k, tk, "content") for k, tk in rows if tk == "hotpot"])):
            want = [c for c in want if F.place[(c[0], c[1], "cut", c[2], "residual")]["mark"]]
            m = re.search(r"Outside the panel's range, " + re.escape(label) + r": (.*?)\.(?= Outside|$)", t)
            if not m:
                if want:
                    self.fail(ln, f"outside-range list ({label}) not printed")
                continue
            items = list(re.finditer(NAME_RE + " " + TASK_RE + r" (strict|content) " + V + r" \(panel " + V + " to "
                                     + V + r"\)", m.group(1)))
            got = [(KEY_OF[i.group(1)], TASK_ROW[i.group(2)], i.group(3)) for i in items]
            self.T.text(f"{ln.where}: outside-range cells ({label})", want, got)
            for i, (k, tk, s) in zip(items, got):
                p = F.place[(k, tk, "cut", s, "residual")]
                self.p(ln, i, 4, p["value"], f"{k} {tk} {s} residual", m.start(1))
                self.p(ln, i, 5, p["min"], f"{k} {tk} {s} panel min", m.start(1))
                self.p(ln, i, 6, p["max"], f"{k} {tk} {s} panel max", m.start(1))

    def readable(self, ln: Line) -> None:
        """Requested arms under 90% readable, with the answer format's share (strict minus content, as a share of the
        strict value, discovery draws) of strict CoT − requested and of the strict residual, and the counts of cells
        where it is more than half."""
        t, F = ln.text, self.F
        m = re.match(r"^Requested replies the strict rule cannot read, discovery draws \(under " + V + r"% readable\), "
                     r"with the answer format's share of strict CoT − requested and of the strict residual \(strict "
                     r"minus content, as a share of the strict value, discovery draws\): (.*?)\. The format is more "
                     r"than half of the strict residual in (\d+) of (\d+) cells and more than half of strict CoT − "
                     r"requested in (\d+) of (\d+); in these cells the content columns carry the comparison\.$", t)
        if not m:
            return self.fail(ln, "readable sentence not parsed")
        self.p(ln, m, 1, READABLE_BELOW, "readable threshold")
        items = list(re.finditer(NAME_RE + " " + TASK_RE + " " + V + r"% readable, content − strict on the requested "
                                 r"arm " + V + r" pp, format (?:" + V + r"%|(– \(strict value 0\))) of CoT − requested "
                                 r"and (?:" + V + r"%|(– \(strict value 0\))) of the residual", m.group(2)))
        want = [(x["key"], x["task"]) for x in F.shares]
        got = [(KEY_OF[i.group(1)], TASK_ROW[i.group(2)]) for i in items]
        self.T.text(f"{ln.where}: low-readability cells", want, got)
        for i, x in zip(items, F.shares):
            k, tk = x["key"], x["task"]
            self.p(ln, i, 3, x["readable"], f"{k} {tk} readable (discovery)", m.start(2))
            self.p(ln, i, 4, x["gap"], f"{k} {tk} content − strict (discovery)", m.start(2))
            for gn, gd, v, what in ((5, 6, x["share_gain"], "CoT − requested"), (7, 8, x["share_residual"],
                                                                                  "the residual")):
                if i.group(gn) is not None:
                    self.p(ln, i, gn, v, f"{k} {tk} format share of {what}", m.start(2))
                else:
                    self.T.text(f"{ln.where}: {k} {tk} format share of {what} (strict value 0)", None, v)
                    ln.use(m.start(2) + i.start(gd), m.start(2) + i.end(gd))
        n = len(F.shares)
        self.p(ln, m, 3, sum(x["share_residual"] is not None and x["share_residual"] > 50 for x in F.shares),
               "cells: format more than half of the residual")
        self.p(ln, m, 4, n, "cells")
        self.p(ln, m, 5, sum(x["share_gain"] is not None and x["share_gain"] > 50 for x in F.shares),
               "cells: format more than half of CoT − requested")
        self.p(ln, m, 6, n, "cells")

    def ds_size(self, ln: Line) -> None:
        """The DeepSeek line: outside the families, and whether the recorded sizes make it the top of the size range."""
        t, F = ln.text, self.F
        m = re.match(r"^DeepSeek V4\.1-Flash-1017 is a single model outside the three families, with no smaller "
                     r"DeepSeek of its release; it carries no family trend\. It is listed as "
                     r"the top of the size range; its published size is (.*? \(.*?https?://\S+?\))(, .*)$", t)
        if not m:
            return self.fail(ln, "DeepSeek line not parsed")
        o = F.sizes[DS]
        # its own published size (V4.1-Flash's, taken for the 1017 checkpoint), numbers and URL
        self.w(ln, m, 1, ds_size_phrase(o), "DeepSeek: its published size, caveat and page")
        others = {k: v for k, v in F.sizes.items() if k in FILES and k != DS}
        tot = max(((Fraction(v["total"]), k) for k, v in others.items() if v.get("total")), default=None)
        act = max(((Fraction(v["active"]), k) for k, v in others.items() if v.get("active")), default=None)
        want = []
        if tot and o.get("total") and tot[0] > Fraction(o["total"]):
            want.append(f"below {NAMES[tot[1]][1]} in total parameters ({others[tot[1]]['total']}B)")
        if act and largest_active(o) is not None and act[0] > largest_active(o):
            want.append(f"below {NAMES[act[1]][1]} in active parameters ({others[act[1]]['active']}B, "
                        f"{others[act[1]]['arch']})")
        rest = (", " + " and ".join(want) + ", so the records do not support that." if want else
                ", the largest recorded in total and in active parameters.")
        self.w(ln, m, 2, rest, "DeepSeek against the largest recorded sizes")

    def bbh0(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.match(r"^BIG-Bench Hard zero-shot forced arm, content − strict, discovery draws, value \(b/M\) against the "
                     r"panel: (.*)\. The panel's runs from " + V + " to " + V + r" \(median " + V + r", (\d+) models\)\.$",
                     t)
        if not m:
            return self.fail(ln, "BIG-Bench Hard zero-shot sentence not parsed")
        pg = place(None, F.bbh_panel_gap)
        self.p(ln, m, 2, pg["min"], "panel gap min")
        self.p(ln, m, 3, pg["max"], "panel gap max")
        self.p(ln, m, 4, pg["median"], "panel gap median")
        self.p(ln, m, 5, pg["n"], "panel gap n")
        items = list(re.finditer(NAME_RE + r" (" + CELL_PAT + r")(?=; |$)", m.group(1)))
        self.T.text(f"{ln.where}: models", list(F.bbh_gap), [KEY_OF[i.group(1)] for i in items])
        for i in items:
            k = KEY_OF[i.group(1)]
            self.cell(f"{ln.where}: {k} BIG-Bench Hard zero-shot forced gap", i.group(2), F.bbh_gap.get(k))
            ln.use(m.start(1) + i.start(2), m.start(1) + i.end(2))

    def bbh5(self, ln: Line) -> None:
        """The five-shot forced gap over all rows of the arm, each model placed against the panel's gaps on that basis
        (results/ladder/mistral_bbh_forced_check.json); the basis claim, the cells, the panel's range and median."""
        t, F = ln.text, self.F
        fp = F.fs_panel
        if fp is None:
            m = re.match(r"^BIG-Bench Hard five-shot forced arm, content − strict, all rows of the arm \(no panel file "
                         r"here, `[^`]+`\): (.*?)\. Strict and content", t)
            if not m:
                return self.fail(ln, "BIG-Bench Hard five-shot sentence (no panel file) not parsed")
            items = list(re.finditer(NAME_RE + " " + V + " pp", m.group(1)))
            self.T.text(f"{ln.where}: models", list(F.bbh_gap), [KEY_OF[i.group(1)] for i in items])
            for i in items:
                k = KEY_OF[i.group(1)]
                self.p(ln, i, 2, F.x[k]["cut"]["bbh"]["gap_all_rows"]["fs_forced"], f"{k} five-shot forced gap",
                       m.start(1))
            return
        m = re.match(r"^BIG-Bench Hard five-shot forced arm, content − strict, all rows of the arm, value \(b/M\) "
                     r"against the panel's gaps on that basis \(([^;)]*(?:; caps not checked here)?); `([^`]+)`, "
                     r"panel values to " + V + r" pp\): (.*?)\. The panel's runs from " + V + " to " + V
                     + r" \(median " + V + r", (\d+) models\)\.(?: Within the panel values' rounding of a panel gap, "
                     r"so the count is not settled by the file: (.*?)\.)? Strict and content are reported side by "
                     r"side", t)
        if not m:
            return self.fail(ln, "BIG-Bench Hard five-shot sentence not parsed")
        fails = [w for w, ok in (("rows", fp["rows_ok"]), ("scorer", fp["scorer_ok"]), ("caps", fp["caps_ok"]))
                 if ok is False]
        want = ("the same rows, scorer and caps" if not fails and fp["caps_ok"] else
                "the same rows and scorer; caps not checked here" if not fails else
                "basis check failed: " + ", ".join(fails))
        self.w(ln, m, 1, want, "five-shot basis")
        self.w(ln, m, 2, FEWSHOT_PANEL, "five-shot panel source")
        self.p(ln, m, 3, Fraction(1, 100), "five-shot panel precision")
        items = list(re.finditer(NAME_RE + r" (" + CELL_PAT + r")(?=; |$)", m.group(4)))
        self.T.text(f"{ln.where}: models", list(F.fs_place), [KEY_OF[i.group(1)] for i in items])
        for i in items:
            k = KEY_OF[i.group(1)]
            self.cell(f"{ln.where}: {k} BIG-Bench Hard five-shot forced gap", i.group(2), F.fs_place.get(k))
            ln.use(m.start(4) + i.start(2), m.start(4) + i.end(2))
        pg = place(None, [x["gap"] for x in fp["gaps"].values()])
        self.p(ln, m, 5, pg["min"], "five-shot panel gap min")
        self.p(ln, m, 6, pg["max"], "five-shot panel gap max")
        self.p(ln, m, 7, pg["median"], "five-shot panel gap median")
        self.p(ln, m, 8, pg["n"], "five-shot panel gap n")
        uns = [k for k in F.fs_place if F.fs_unsettled.get(k)]
        want_uns = "; ".join(f"{NAMES[k][1]} ({', '.join(F.fs_unsettled[k])})" for k in uns) or None
        if m.group(9) is not None:
            self.w(ln, m, 9, want_uns, "five-shot counts not settled by the file")
        elif uns:
            self.fail(ln, f"five-shot: counts not settled by the file are not listed ({uns})")

    def order_words(self) -> str:
        """'n pairs of ladder models change order' and the ties that separate, counted apart (rules.ties)."""
        F = self.F
        txt = f"{F.flips} pairs of ladder models change order" + without_readable([a["score"] for a, _ in F.flip_list])
        for r, other in (("cut", "uncut"), ("uncut", "cut")):
            n = sum(x[2] == r for x in F.tie_list)
            if n:
                txt += (f"; {n} pair{'s' if n > 1 else ''} tied in the {r} reading separate{'' if n > 1 else 's'} in "
                        f"the {other}")
        return txt

    def cap(self, ln: Line) -> None:
        """Section 1: the findings that depend on the cap, each recomputed here with the uncut values on its own basis:
        the residual band, the marks of every placed, printed quantity, the side of the panel median of CoT − forced
        (content) and the requested arm's readable share against 90%."""
        t, F = ln.text, self.F
        m0 = re.match(r"^Findings above that depend on the cap \((all on " + TASK_RE + r"|on (?:" + TASK_RE[1:-1]
                      + r")(?:, (?:" + TASK_RE[1:-1] + r"))*|none)\), ", t)
        if not m0:
            return self.fail(ln, "cap-dependence sentence: its benchmarks not parsed")
        ln.use(*m0.span(1))
        t = "Findings above that depend on the cap, " + t[m0.end():]
        sh = m0.end() - len("Findings above that depend on the cap, ")
        ln = _Shifted(ln, sh)
        m = re.match(r"^Findings above that depend on the cap, each recomputed with the uncut values on its own basis\. "
                     r"Residual within ±" + V + r" pp: (.*?)\. Marks ↑ or ↓ \(outside the panel's range\) that change: "
                     r"(.*?)\. Side of the panel median of CoT − forced \(content\): (.*?)\. Requested replies under "
                     + V + r"% readable, discovery draws: (.*?)\. Pairs of ladder models that change order, placements "
                     r"that move and family directions that change: the two lines above and section 9\.$", t)
        if not m:
            return self.fail(ln, "cap-dependence sentence not parsed")
        self.p(ln, m, 1, RESIDUAL_BAND, "cap: residual band")
        # the residual band, per basis
        cells = {b: [(k, tk) for k in F.present for tk in TASKS if basis_of(tk, "content") == b
                     and F.place.get((k, tk, "cut", "content", "residual"))
                     and F.place.get((k, tk, "uncut", "content", "residual"))] for b in ("discovery", "pooled")}

        def res(k, tk, r):
            return F.place[(k, tk, r, "content", "residual")]["value"]

        def within(c, r):
            v = res(*c, r)
            return v is not None and abs(v) <= RESIDUAL_BAND
        o2 = m.start(2)
        parts = list(re.finditer(r"(content, discovery draws|HotpotQA content, pooled rows): (?:(\d+) of (\d+) in "
                                 r"both readings|(\d+) of (\d+) cut, (\d+) of (\d+) uncut(?: \(([^)]*)\))?)",
                                 m.group(2)))
        bases = ["discovery" if x.group(1).startswith("content") else "pooled" for x in parts]
        self.T.text(f"{ln.where}: cap: residual band, bases", [b for b in ("discovery", "pooled") if cells[b]], bases)
        for x, b in zip(parts, bases):
            cs = cells[b]
            wc, wu = [c for c in cs if within(c, "cut")], [c for c in cs if within(c, "uncut")]
            self.T.text(f"{ln.where}: cap: band {b}, form", "both" if wc == wu else "split",
                        "both" if x.group(2) is not None else "split")
            if x.group(2) is not None:
                self.p(ln, x, 2, len(wc), f"cap: band {b}, within", o2)
                self.p(ln, x, 3, len(cs), f"cap: band {b}, cells", o2)
                continue
            self.p(ln, x, 4, len(wc), f"cap: band {b}, within cut", o2)
            self.p(ln, x, 5, len(cs), f"cap: band {b}, cells", o2)
            self.p(ln, x, 6, len(wu), f"cap: band {b}, within uncut", o2)
            self.p(ln, x, 7, len(cs), f"cap: band {b}, cells", o2)
            o8 = o2 + (x.start(8) if x.group(8) is not None else 0)
            items = list(re.finditer(r"(leaves|enters): " + NAME_RE + " " + TASK_RE + ", " + V + " cut, " + V
                                     + " uncut", x.group(8) or ""))
            want = sorted([("leaves",) + c for c in wc if c not in wu] + [("enters",) + c for c in wu if c not in wc])
            got = sorted((i.group(1), KEY_OF[i.group(2)], TASK_ROW[i.group(3)]) for i in items)
            self.T.text(f"{ln.where}: cap: band {b}, cells that enter or leave", want, got)
            for i in items:
                k, tk = KEY_OF[i.group(2)], TASK_ROW[i.group(3)]
                self.p(ln, i, 4, res(k, tk, "cut"), f"cap: {k} {tk} residual cut", o8)
                self.p(ln, i, 5, res(k, tk, "uncut"), f"cap: {k} {tk} residual uncut", o8)
        # marks that change, over every placed, printed quantity
        want = [(k, tk, sc, q) for k in F.present for tk, sc, q in F.printed["placed"]
                if F.place.get((k, tk, "cut", sc, q)) and F.place.get((k, tk, "uncut", sc, q))
                and F.place[(k, tk, "cut", sc, q)]["mark"] != F.place[(k, tk, "uncut", sc, q)]["mark"]]
        o3 = m.start(3)
        items = list(re.finditer(NAME_RE + " " + TASK_RE + r" (strict|content|readable) " + QLAB_RE
                                 + r", (discovery draws|pooled rows): (" + CELL_PAT + r") cut, (" + CELL_PAT
                                 + r") uncut", m.group(3)))
        got = [(KEY_OF[i.group(1)], TASK_ROW[i.group(2)], i.group(3), Q_OF[i.group(4)]) for i in items]
        self.T.text(f"{ln.where}: cap: marks that change", sorted(want), sorted(got))
        self._cap_marks = want
        if not items:
            self.w(ln, m, 3, "none", "cap: marks that change")
        for i, (k, tk, sc, q) in zip(items, got):
            self.T.text(f"{ln.where}: cap: {k} {tk} {sc} {q} basis",
                        BASIS_WORDS["discovery" if sc == "readable" else basis_of(tk, sc)], i.group(5))
            for gi, r in ((6, "cut"), (7, "uncut")):
                self.cell(f"{ln.where}: cap: {k} {tk} {sc} {q} {r}", i.group(gi), F.place.get((k, tk, r, sc, q)))
            ln.use(o3 + i.start(), o3 + i.end())
        # the side of the panel median of CoT − forced (content)

        def above(p):
            return p["value"] is not None and p["value"] > p["median"] + TOL
        want = [(k, tk) for k in F.present for tk in TASKS
                if F.place.get((k, tk, "cut", "content", "cot_minus_forced"))
                and F.place.get((k, tk, "uncut", "content", "cot_minus_forced"))
                and above(F.place[(k, tk, "cut", "content", "cot_minus_forced")])
                != above(F.place[(k, tk, "uncut", "content", "cot_minus_forced")])]
        o4 = m.start(4)
        items = list(re.finditer(NAME_RE + " " + TASK_RE + r", (discovery draws|pooled rows): (" + CELL_PAT
                                 + r") cut, (" + CELL_PAT + r") uncut", m.group(4)))
        got = [(KEY_OF[i.group(1)], TASK_ROW[i.group(2)]) for i in items]
        self.T.text(f"{ln.where}: cap: median sides that change", want, got)
        self._cap_median = want
        if not items:
            self.w(ln, m, 4, "the same in both readings", "cap: median sides")
        for i, (k, tk) in zip(items, got):
            self.T.text(f"{ln.where}: cap: {k} {tk} median basis", BASIS_WORDS[basis_of(tk, "content")], i.group(3))
            for gi, r in ((4, "cut"), (5, "uncut")):
                self.cell(f"{ln.where}: cap: {k} {tk} median {r}", i.group(gi),
                          F.place.get((k, tk, r, "content", "cot_minus_forced")))
            ln.use(o4 + i.start(), o4 + i.end())
        # the requested arm's readable share against 90%, discovery draws
        self.p(ln, m, 5, READABLE_BELOW, "cap: readable threshold")
        low = {r: [(k, tk) for k in F.present for tk in TASKS if tk in F.x[k][r]
                   and F.place.get((k, tk, "cut", "content", "cot_minus_forced"))
                   and g(F.x[k][r][tk], "readable", "requested", "discovery") is not None
                   and g(F.x[k][r][tk], "readable", "requested", "discovery") < READABLE_BELOW] for r in READINGS}
        mm = re.match(r"^the same (\d+) cells in both readings$", m.group(6))
        self.T.text(f"{ln.where}: cap: readable cells, form", low["cut"] == low["uncut"], mm is not None)
        if mm:
            self.p(ln, mm, 1, len(low["cut"]), "cap: readable cells", m.start(6))
        else:
            mm = re.match(r"^(\d+) cells cut, (\d+) uncut \((.*)\)$", m.group(6))
            if not mm:
                return self.fail(ln, "cap: readable cells not parsed")
            self.p(ln, mm, 1, len(low["cut"]), "cap: readable cells cut", m.start(6))
            self.p(ln, mm, 2, len(low["uncut"]), "cap: readable cells uncut", m.start(6))
            got = sorted((i.group(1), KEY_OF[i.group(2)], TASK_ROW[i.group(3)]) for i in
                         re.finditer(r"only (cut|uncut): " + NAME_RE + " " + TASK_RE, mm.group(3)))
            want = sorted([("cut",) + c for c in low["cut"] if c not in low["uncut"]]
                          + [("uncut",) + c for c in low["uncut"] if c not in low["cut"]])
            self.T.text(f"{ln.where}: cap: readable cells in one reading only", want, got)
        # the benchmarks the changes are on, as the sentence opens
        moved = {tk for b in ("discovery", "pooled") for k, tk in cells[b] if within((k, tk), "cut")
                 != within((k, tk), "uncut")}
        moved |= {x[1] for x in self._cap_marks} | {x[1] for x in self._cap_median}
        moved |= {tk for k, tk in set(low["cut"]) ^ set(low["uncut"])}
        where = [tk for tk in TASKS if tk in moved]
        want = (f"all on {TASK_NAME[where[0]]}" if len(where) == 1 else
                "on " + ", ".join(TASK_NAME[x] for x in where) if where else "none")
        self.T.text(f"{ln.where}: cap: the benchmarks the changes are on", want, m0.group(1))

    def largest_part(self, ln: Line, m, gv: int, gn: int, gl: int, readable: bool | None, what: str,
                     off: int = 0) -> None:
        """'<v> pp[, shared by <n> quantities] (<Model>, <Benchmark>: <score> <quantity>, ...; ...)': the value, the
        count and the set of quantities that share the largest change, every one of them named (ties reported)."""
        mx, want = largest(self.F, readable)
        self.p(ln, m, gv, mx, f"{what}: value", off)
        if m.group(gn) is not None:
            self.p(ln, m, gn, len(want), f"{what}: quantities that share it", off)
        elif len(want) != 1:
            self.fail(ln, f"{what}: {len(want)} quantities share the largest change and the count is not printed")
        got = set()
        for grp in re.finditer(NAME_RE + ", " + TASK_RE + r": (.*?)(?=; " + NAME_NC + r"|$)", m.group(gl)):
            for it in grp.group(3).split(", "):
                sm = re.match(r"^(strict|content|readable) " + QLAB_RE + "$", it)
                got.add((KEY_OF[grp.group(1)], TASK_ROW[grp.group(2)], sm.group(1), Q_OF[sm.group(2)]) if sm
                        else ("not parsed", it, "", ""))
        self.T.text(f"{ln.where}: {what}: the quantities that share it", sorted(want), sorted(got))
        a, b = m.span(gl)
        ln.use(a + off, b + off)

    LARGEST = V + r" pp(?:, shared by (\d+) quantities)? \(([^()]*)\)"

    def cut_uncut(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.match(r"^Cut against uncut, every placed quantity on the discovery draws "
                     r"\((.*?)\): the largest change is " + self.LARGEST + r"(?:; leaving out the readable shares, "
                     + self.LARGEST + r")?; changes of " + V + r" pp or more: ([^;]*); (.*) \(section 9\)\.$", t)
        if not m:
            return self.fail(ln, "cut-against-uncut sentence not parsed")
        self.w(ln, m, 1, PLACED_WORDS, "what the comparison covers")
        self.T.text(f"{ln.where}: quantities compared (ladder.md's placed rows, per model and benchmark)",
                    sorted({(d["task"], d["score"], d["q"]) for d in F.diffs}), sorted(F.printed["placed"]))
        self.largest_part(ln, m, 2, 3, 4, None, "largest change")
        split = any(x[2] == "readable" for x in largest(F)[1])
        self.T.text(f"{ln.where}: the largest change without the readable shares printed", split,
                    m.group(5) is not None)
        if m.group(5) is not None:
            self.largest_part(ln, m, 5, 6, 7, False, "largest change without the readable shares")
        ds = [d for d in F.diffs if d["d"] is not None]
        self.p(ln, m, 8, CUT_BIG, "change threshold")
        big = sorted({TASK_NAME[d["task"]] for d in ds if abs(d["d"]) >= CUT_BIG})
        self.w(ln, m, 9, ", ".join(big) if big else "none", "benchmarks with big changes")
        self.w(ln, m, 10, self.order_words(), "pairs that change order, and ties that separate")

    def glance(self, where: str, head: list[str], rows: list[list[str]]) -> None:
        F = self.F
        cols = {"CoT − forced, strict": ("strict", "cot_minus_forced"), "CoT − forced, content": ("content",
                                                                                                    "cot_minus_forced"),
                "CoT − requested, strict": ("strict", "cot_minus_requested"),
                "CoT − requested, content": ("content", "cot_minus_requested"), "w": ("strict", "w"),
                "Residual, strict": ("strict", "residual"), "Residual, content": ("content", "residual")}
        hot = head[1] != "Benchmark"
        want_rows = [(k, tk) for k in F.present for tk in TASKS if (tk == "hotpot") == hot
                     and (k, tk, "cut", "strict", "w") in F.place]
        got_rows = [(KEY_OF.get(r[0]), "hotpot" if hot else TASK_ROW.get(r[1])) for r in rows]
        self.T.text(f"{where}: rows", want_rows, got_rows)
        for r, (k, tk) in zip(rows, got_rows):
            if k is None or tk is None:
                continue
            for h, c in zip(head, r):
                base = re.sub(r" \((discovery draws|pooled rows)\)$", "", h)
                bm = re.search(r"\((discovery draws|pooled rows)\)$", h)
                if base in cols:
                    s, q = cols[base]
                    if bm:
                        self.T.text(f"{where}: {h} basis", BASIS_WORDS[basis_of(tk, s)], bm.group(1))
                    self.cell(f"{where}: {k} {tk} {h}", c, F.place[(k, tk, "cut", s, q)])
                elif base in ("Requested readable (%)", "Requested readable, %"):
                    if tk == "hotpot":
                        self.T.printed(f"{where}: {k} {tk} {h}", c, g(F.x[k]["cut"][tk], "readable", "requested",
                                                                     "discovery"))
                    else:
                        self.cell(f"{where}: {k} {tk} {h}", c, F.place[(k, tk, "cut", "readable", "requested")])

    # -- section 5 -----------------------------------------------------------------------------------------------
    def family_table(self, where: str, fam: str, label: str, head: list[str], rows: list[list[str]]) -> None:
        F = self.F
        f = F.fam[fam]
        m = re.match(TASK_RE + r"(?: \(strict / content; discovery draws\)|, (strict|content) \((discovery draws|"
                     r"pooled rows)\)):$", label)
        if not m:
            self.T.n += 1
            self.T.bad.append(f"{where}: table label not parsed: {label!r}")
            return
        tk = TASK_ROW[m.group(1)]
        scores = ("strict", "content") if m.group(2) is None else (m.group(2),)
        if m.group(3):
            self.T.text(f"{where}: {label} basis", BASIS_WORDS[basis_of(tk, m.group(2))], m.group(3))
        else:
            self.T.text(f"{where}: {label} basis", "discovery", basis_of(tk, "content"))
        rung_cols = head[1:-1]
        self.T.text(f"{where}: rung columns", [f"{SIZE[r['key']]} ({r['src']})" for r in f["rungs"]], rung_cols)
        for row in rows:
            q = Q_OF.get(row[0])
            if q is None:
                self.T.n += 1
                self.T.bad.append(f"{where}: quantity label not known: {row[0]!r}")
                continue
            if q == "corrected_gain":
                self.T.text(f"{where}: {tk} corrected-gain label names the pooled product",
                            scores == ("content",) and tk == "hotpot", row[0] == POOLED_CG)
            for r, c in zip(f["rungs"], row[1:-1]):
                path = f"{where}: {fam} {tk} {row[0]} {r['key']}"
                if not r["present"]:
                    self.T.text(path, "waits", c)
                    continue
                if q == "w":
                    self.T.printed(path, c, rung_value(F, r["key"], r["src"], tk, "strict", "w", "cut"))
                    continue
                parts = c.split(" / ")
                if len(parts) != len(scores):
                    self.T.n += 1
                    self.T.bad.append(f"{path}: {len(parts)} values for {len(scores)} scores: {c!r}")
                    continue
                for s, part in zip(scores, parts):
                    self.T.printed(f"{path} {s}", part, rung_value(F, r["key"], r["src"], tk, s, q, "cut"))
            want = (f["dir"][(tk, "strict", "w")] or "–") if q == "w" else \
                " / ".join(f["dir"][(tk, s, q)] or "–" for s in scores)
            self.T.text(f"{where}: {fam} {tk} {row[0]} direction", want, row[-1])

    def context_table(self, where: str, fam: str, rows: list[list[str]]) -> None:
        P = self.F.panel
        want = [(k, tk) for k in FAMILIES[fam]["context"] for tk in TASKS if k in P[tk].index]
        self.T.text(f"{where}: rows", want, [(r[0], TASK_ROW.get(r[1])) for r in rows])
        for r in rows:
            k, tk = r[0], TASK_ROW.get(r[1])
            if tk is None or k not in P[tk].index:
                continue
            self.T.text(f"{where}: {k} {tk} basis", "discovery draws" if tk in DISC_TASKS else
                        "content on pooled rows; strict and w on the discovery draws", r[2])
            for q, c in zip(("forced", "cot", "cot_minus_forced", "cot_minus_requested"), r[3:7]):
                m = re.match(r"^" + V + r" \(" + V + r"\)$", c)
                if not m:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: {k} {tk} {q}: not of the form content (strict): {c!r}")
                    continue
                self.T.printed(f"{where}: {k} {tk} {q} content", m.group(1), P[tk].at[k, f"{q}_content"])
                self.T.printed(f"{where}: {k} {tk} {q} strict", m.group(2), P[tk].at[k, f"{q}_strict"])
            self.T.printed(f"{where}: {k} {tk} w", r[7], P[tk].at[k, "w"])

    # -- section 6 -----------------------------------------------------------------------------------------------
    def placement_table(self, where: str, h3: str, head: list[str], rows: list[list[str]]) -> None:
        F = self.F
        m = re.match(TASK_RE + r"(?:, (strict|content))? \((discovery draws|pooled rows)\)$", h3)
        if not m:
            self.T.n += 1
            self.T.bad.append(f"{where}: heading not parsed: {h3!r}")
            return
        tk = TASK_ROW[m.group(1)]
        keys = [KEY_OF.get(h) for h in head[5:]]
        self.T.text(f"{where}: model columns", [k for k in F.present if (k, tk, "cut", "strict", "w") in F.place],
                    keys)
        for row in rows:
            s = row[0]
            if s == "–":
                a = re.match(r"^Readable, (requested|forced|cot) \(%\)$", row[1])
                if not a:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: row label not parsed: {row[1]!r}")
                    continue
                pls = [F.place.get((k, tk, "cut", "readable", a.group(1))) for k in keys]
                what = f"readable {a.group(1)}"
                basis = "discovery"
            else:
                q = Q_OF.get(row[1])
                if q is None:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: quantity label not known: {row[1]!r}")
                    continue
                pls = [F.place.get((k, tk, "cut", s, q)) for k in keys]
                what = f"{s} {q}"
                basis = basis_of(tk, s)
                lab_pooled = row[1] in (POOLED_CG, f"CoT − requested minus {POOLED_CG}")
                if q in ("corrected_gain", "residual"):
                    self.T.text(f"{where}: {what} label names the pooled product", basis == "pooled", lab_pooled)
            self.T.text(f"{where}: {what} basis in heading", BASIS_WORDS[basis], m.group(3))
            if not pls or pls[0] is None:
                continue
            for i, nm in enumerate(("min", "median", "max")):
                self.T.printed(f"{where}: {tk} {what} panel {nm}", row[2 + i], pls[0][nm])
            for k, pl, c in zip(keys, pls, row[5:]):
                self.cell(f"{where}: {k} {tk} {what}", c, pl)

    # -- sections 7 to 9: per-model tables -------------------------------------------------------------------------
    def model_table(self, where: str, reading: str, head: list[str], rows: list[list[str]]) -> None:
        """Sections 7 and 8: per model, benchmark and score, every column on its named basis."""
        F = self.F
        want = [(k, tk, s) for k in F.present for tk in TASKS if tk in F.x[k][reading] for s in SCORE_COL]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1]), r[2]) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        for r, (k, tk, s) in zip(rows, got):
            if k is None or tk is None or s not in SCORE_COL:
                continue
            x = F.x[k][reading][tk]
            for h, c in zip(head[3:], r[3:]):
                hm = re.match(r"^(.*) \((48 draws|discovery draws)\)$", h)
                if not hm or hm.group(1) not in Q_OF:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: column not parsed: {h!r}")
                    continue
                basis = "all_draws" if hm.group(2) == "48 draws" else "discovery"
                self.T.printed(f"{where}: {k} {tk} {s} {reading} {h}", c, value(x, s, basis, Q_OF[hm.group(1)]))

    def fewshot_table(self, where: str, rows: list[list[str]]) -> None:
        F = self.F
        want = [(k, tk, arm, rd) for k in F.present for tk in TASKS if tk in F.x[k]["cut"] for rd in READINGS
                for arm in (F.x[k][rd].get(tk, {}).get("fewshot") or {})]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1]), r[2], r[3]) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        for r, (k, tk, arm, rd) in zip(rows, got):
            d = g(F.x, k, rd, tk, "fewshot", arm)
            if d is None:
                continue
            sh = d.get("share_longer_than_forced_cap")
            for c, v, what in ((r[4], d["rows"], "rows"), (r[5], d["configs"], "configurations"),
                               (r[6], d["strict_pp_config_mean"], "strict"), (r[7], d["content_pp_config_mean"], "content"),
                               (r[8], d.get("strict_pp_all_draws"), "strict (draws)"),
                               (r[9], d.get("content_pp_all_draws"), "content (draws)"),
                               (r[10], None if sh is None else 100 * sh, "share longer than the forced cap")):
                self.T.printed(f"{where}: {k} {tk} {arm} {rd} {what}", c, v)

    def readable_table(self, where: str, rows: list[list[str]]) -> None:
        F = self.F
        want = [(k, tk) for k in F.present for tk in TASKS if tk in F.x[k]["cut"]]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1])) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        arms = ("requested", "forced", "cot", "fs_forced", "fs_requested")
        for r, (k, tk) in zip(rows, got):
            if k is None or tk is None:
                continue
            x = F.x[k]["cut"][tk]
            for a, c in zip(arms, r[2:7]):
                self.T.printed(f"{where}: {k} {tk} readable {a}", c, g(x, "readable", a, "all_rows"))
            for a, c in zip(arms, r[7:12]):
                self.T.printed(f"{where}: {k} {tk} content − strict {a}", c, x["gap_all_rows"].get(a))

    def cut_table(self, where: str, rows: list[list[str]]) -> None:
        F = self.F
        raw = {v: a for a, v in LABEL.items()}
        raw["requested"], raw["cot"] = "direct_answer", "zero_shot_cot"
        want = [(k, tk, a) for k in F.present for tk in TASKS if tk in F.x[k]["cut"]
                for a in ("requested", "cot", "fs_requested") if a in F.x[k]["cut"][tk].get("cut_at_panel_cap", {})]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1]), r[2]) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        for r, (k, tk, a) in zip(rows, got):
            x, u = g(F.x, k, "cut", tk), g(F.x, k, "uncut", tk)
            if x is None:
                continue
            n, c = x["rows_by_arm"].get(raw.get(a)), x["cut_at_panel_cap"].get(a)
            self.T.printed(f"{where}: {k} {tk} {a} rows", r[3], n)
            self.T.printed(f"{where}: {k} {tk} {a} cut at the panel's cap", r[4], c)
            self.T.printed(f"{where}: {k} {tk} {a} share cut", r[5], None if not n else Fraction(100 * c, n))
            self.T.printed(f"{where}: {k} {tk} {a} ended at the raised cap", r[6],
                           None if u is None else u.get("finish_length", {}).get(a))

    def change_table(self, where: str, rows: list[list[str]], count_line: str | None) -> None:
        F = self.F
        big = [d for d in F.diffs if d["d"] is not None and abs(d["d"]) >= Fraction(1, 20)]
        m = re.search(r"\((\d+) of (\d+); (\d+) change at all;", count_line or "")
        if not m:
            self.T.n += 1
            self.T.bad.append(f"{where}: the count line of the changes table was not found")
        else:
            self.T.printed(f"{where}: changes of 0.05 pp or more", m.group(1), len(big))
            self.T.printed(f"{where}: differences", m.group(2), len(F.diffs))
            self.T.printed(f"{where}: changes at all", m.group(3),
                           sum(1 for d in F.diffs if d["d"] is not None and abs(d["d"]) > TOL))
        key = {(d["model"], d["task"], d["score"], d["q"]): d for d in big}
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1]), r[2], Q_OF.get(r[3])) for r in rows]
        self.T.text(f"{where}: rows (as a set)", sorted(key, key=str), sorted(got, key=str))
        for r, kk in zip(rows, got):
            d = key.get(kk)
            if d is None:
                continue
            for c, v, what in ((r[4], d["cut"], "cut"), (r[5], d["uncut"], "uncut"), (r[6], d["d"], "uncut − cut")):
                self.T.printed(f"{where}: {kk} {what}", c, v)

    # -- section 9: family directions under the uncut reading --------------------------------------------------
    def sec9_line(self, ln: Line) -> None:
        """Section 9 outside its tables: the change-table count line, the reversals, the ties that separate, the
        placements that move and the family directions that change, each count and each listed line."""
        t, F = ln.text, self.F
        m = re.match(r"^Quantities that change by (0\.05) pp or more between the readings, every placed quantity on the "
                     r"discovery draws \((\d+) of (\d+); (\d+) change at all; of these, by (0\.05) pp or more and at "
                     r"all: residuals (\d+) and (\d+) of (\d+); readable shares (\d+) and (\d+) of (\d+); every change "
                     r"is in `ladder\.json`\)\. The placed quantities are section 6's rows; Score readable is the share of "
                     r"replies the strict rule can read, per zero-shot arm \(%\), placed on GSM8K, MATH-500 and "
                     r"BIG-Bench Hard, where the panel has it:$", t)
        if m:
            def counts(ds):
                return (sum(1 for d in ds if d["d"] is not None and abs(d["d"]) >= Fraction(1, 20)),
                        sum(1 for d in ds if d["d"] is not None and abs(d["d"]) > TOL), len(ds))
            self.p(ln, m, 1, Fraction(1, 20), "listing threshold")
            a, b, n = counts(F.diffs)
            self.p(ln, m, 2, a, "changes of 0.05 pp or more")
            self.p(ln, m, 3, n, "differences")
            self.p(ln, m, 4, b, "changes at all")
            self.p(ln, m, 5, Fraction(1, 20), "listing threshold")
            for gi, what, pred in ((6, "residuals", lambda d: d["q"] == "residual"),
                                   (9, "readable shares", lambda d: d["score"] == "readable")):
                for j, v in enumerate(counts([d for d in F.diffs if pred(d)])):
                    self.p(ln, m, gi + j, v, f"{what}: {('0.05 pp or more', 'at all', 'compared')[j]}")
            return
        m = re.match(r"^Pairs of ladder models whose order reverses between the readings \(each reading orders the "
                     r"pair, the two in opposite ways\), discovery draws: (\d+)( \((\d+) without the readable "
                     r"shares\))?\.$", t)
        if m:
            self.p(ln, m, 1, F.flips, "reversals")
            sc = [a["score"] for a, _ in F.flip_list]
            self.T.text(f"{ln.where}: reversals without the readable shares printed", without_readable(sc) != "",
                        m.group(2) is not None)
            if m.group(3) is not None:
                self.p(ln, m, 3, sum(1 for x in sc if x != "readable"), "reversals without the readable shares")
            self.s9["flip_count"] = True
            return
        m = re.match(r"^Pairs of ladder models tied exactly in one reading that separate in the other \(a tie, not a "
                     r"reversal\), discovery draws: (\d+)\.$", t)
        if m:
            self.p(ln, m, 1, len(F.tie_list), "ties that separate")
            self.s9["tie_count"] = True
            return
        m = re.match(r"^- " + TASK_RE + r", (strict|content|readable), " + QLAB_RE + ": " + NAME_RE + " and " + NAME_RE
                     + r", cut " + V + " and " + V + ", uncut " + V + " and " + V + r"(\.|; tied in the (cut|uncut) "
                     r"reading\.)$", t)
        if m:
            key = (TASK_ROW[m.group(1)], m.group(2), Q_OF[m.group(3)], KEY_OF[m.group(4)], KEY_OF[m.group(5)])
            pool = F.tie_list if m.group(10) != "." else F.flip_list
            hit = next((x for x in pool if (x[0]["task"], x[0]["score"], x[0]["q"], x[0]["model"], x[1]["model"])
                        == key), None)
            if hit is None:
                return self.fail(ln, "pair not among the recomputed " + ("ties" if m.group(10) != "." else
                                                                         "reversals"))
            a, b = hit[0], hit[1]
            for gi, v in ((6, a["cut"]), (7, b["cut"]), (8, a["uncut"]), (9, b["uncut"])):
                self.p(ln, m, gi, v, f"pair value {gi}")
            if m.group(11):
                self.w(ln, m, 11, hit[2], "the reading that ties the pair")
            self.s9["ties" if m.group(10) != "." else "flips"].append(key)
            return
        m = re.match(r"^Placements against the panel that change when the uncut value is used, each on the panel's "
                     r"basis \(named per line\): (\d+)( \((\d+) without the readable shares\))?\.$", t)
        if m:
            self.p(ln, m, 1, len(F.rank_list), "placements that move")
            sc = [x[2] for x in F.rank_list]
            self.T.text(f"{ln.where}: placements without the readable shares printed", without_readable(sc) != "",
                        m.group(2) is not None)
            if m.group(3) is not None:
                self.p(ln, m, 3, sum(1 for x in sc if x != "readable"), "placements without the readable shares")
            self.s9["rank_count"] = True
            return
        m = re.match(r"^- " + NAME_RE + ", " + TASK_RE + r", (strict|content|readable), " + QLAB_RE + r", (discovery draws|pooled "
                     r"rows): above (\d+) of (\d+)(?:, (\d+) tied)? \(cut\); above (\d+) of (\d+)(?:, (\d+) tied)? "
                     r"\(uncut\)\.$", t)
        if m:
            key = (KEY_OF[m.group(1)], TASK_ROW[m.group(2)], m.group(3), Q_OF[m.group(4)])
            hit = next((x for x in F.rank_list if x[:4] == key), None)
            if hit is None:
                return self.fail(ln, "placement move not among the recomputed")
            a, b = hit[4], hit[5]
            self.w(ln, m, 5, BASIS_WORDS[basis_of(key[1], key[2])], "basis")
            self.p(ln, m, 6, a["below"], "cut below")
            self.p(ln, m, 7, a["n"], "cut panel n")
            self.tie(ln, m, 8, a, "cut")
            self.p(ln, m, 9, b["below"], "uncut below")
            self.p(ln, m, 10, b["n"], "uncut panel n")
            self.tie(ln, m, 11, b, "uncut")
            self.s9["ranks"].append(key)
            return
        m = re.match(r"^Family directions \(section 5\) that change when the ladder rungs' uncut values are used, "
                     r"each on the basis of its section 5 table \(named per line\): (\d+)\.$", t)
        if m:
            self.p(ln, m, 1, len(F.fam_changes), "family direction changes")
            self.s9["fam_count"] = True
            return
        m = re.match(r"^- " + FAM_RE + ", " + TASK_RE + r", (strict|content), " + QLAB_RE + r", (discovery draws|"
                     r"pooled rows): cut " + VALS + r" \(" + DIRW + r"\), uncut " + VALS + r" \(" + DIRW + r"\)\.$", t)
        if m:
            key = (m.group(1), TASK_ROW[m.group(2)], m.group(3), Q_OF[m.group(4)])
            x = next((x for x in F.fam_changes if (x["family"], x["task"], x["score"], x["q"]) == key), None)
            if x is None:
                return self.fail(ln, "family direction change not among the recomputed")
            self.w(ln, m, 5, BASIS_WORDS[x["basis"]], "basis")
            self.vals(ln, m, 6, x["cut"], "cut values")
            self.w(ln, m, 7, x["dc"], "cut direction")
            self.vals(ln, m, 8, x["uncut"], "uncut values")
            self.w(ln, m, 9, x["du"], "uncut direction")
            self.s9["fams"].append(key)
            return

    def sec9_done(self) -> None:
        F, s9 = self.F, self.s9
        for flag, what in (("flip_count", "reversal count"), ("tie_count", "tie count"), ("rank_count",
                                                                                          "placement-move count"),
                           ("fam_count", "family-direction count")):
            if not s9.get(flag):
                self.T.n += 1
                self.T.bad.append(f"ladder.md section 9: the {what} line was not found")
        self.T.text("ladder.md section 9: reversals listed", sorted(
            (a["task"], a["score"], a["q"], a["model"], b["model"]) for a, b in F.flip_list), sorted(s9["flips"]))
        self.T.text("ladder.md section 9: ties listed", sorted(
            (a["task"], a["score"], a["q"], a["model"], b["model"]) for a, b, _ in F.tie_list), sorted(s9["ties"]))
        self.T.text("ladder.md section 9: placement moves listed", sorted(x[:4] for x in F.rank_list), sorted(s9["ranks"]))
        self.T.text("ladder.md section 9: family direction changes listed",
                    sorted((x["family"], x["task"], x["score"], x["q"]) for x in F.fam_changes), sorted(s9["fams"]))

    def footnote(self, ln: Line) -> None:
        t = ln.text
        if t.startswith("F1."):
            return self.f1(ln)
        if t.startswith("F2."):
            return self.f2(ln)
        if t.startswith("F3."):
            m = re.search(r"The panel ran in (.*) \(`config/models\.yaml`, `config/models_extension\.yaml`\)\.$", t)
            if not m:
                return self.fail(ln, "F3: the panel's precision not parsed")
            self.w(ln, m, 1, self.precision_phrase(), "F3: the panel's precision")
            return
        if t.startswith("F8."):
            return self.f8(ln)
        if t.startswith("F4."):
            for m in re.finditer(NAME_RE + r" (batch-invariant|default \(-dk fallback\)|the provider's)", t):
                k = KEY_OF[m.group(1)]
                self.w(ln, m, 2, kernels(self.F.root, k), f"{k} kernels")
            return
        if t.startswith("F6."):
            m = re.search(r"pooled over every row of the (\d+) configurations", t)
            if m:
                self.p(ln, m, 1, self.F.hotpot_configs, "HotpotQA configurations")
            for m in re.finditer(r"\b(\d+)-draw field", t):
                self.p(ln, m, 1, self.F.n_all, "draws")
            return
        if t.startswith("F7."):
            return self.f7(ln)

    def f7(self, ln: Line) -> None:
        """F7: Qwen3.5-122B-A10B's pilot check, against the check's own files: the verdict quoted from the .md (its
        bold line) and the .json, the rule's threshold, the requested arms' boxed counts per arm and benchmark and their
        pooled share, and the run key the full run used."""
        t, root = ln.text, self.F.root
        pj_path, md_path = root / PILOT_QWEN, root / PILOT_MD
        if not pj_path.exists() or not md_path.exists():
            return self.fail(ln, "F7: the pilot check's files are not there")
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        m = re.match(r"^F7\. Qwen3\.5-122B-A10B failed its pilot check \(`" + re.escape(PILOT_MD) + r"`: \"([^\"]*)\"\)\. "
                     r"The rule asks for a boxed answer in at least (\d+)% of the requested arms' rows \(zero- and "
                     r"five-shot, pooled over the benchmarks; `scripts/check_ladder_pilot\.py`\); the pilot gave one in "
                     r"(\d+) of (\d+) \(" + V + r"\): (.*)\. The full run went ahead under the `([^`]+)` key \(default "
                     r"kernels, the panel's Qwen3\.5 mode\) with no recorded release decision, and its outputs are "
                     r"scored as they are\.$", t)
        if not m:
            return self.fail(ln, "F7 not parsed")
        verdict = str(pj.get("verdict", ""))
        self.T.text(f"{ln.where}: F7: the check failed", True, verdict.startswith("PILOT FAIL"))
        self.w(ln, m, 1, verdict, "F7 quotes the pilot check's verdict (json)")
        self.T.text(f"{ln.where}: F7: the quote is the .md's bold verdict line", True,
                    f"**{m.group(1)}**" in md_path.read_text(encoding="utf-8").splitlines())
        self.p(ln, m, 2, PILOT_MIN, "F7: the rule's threshold")
        req = {(x["task"], x["arm"]): (int(x["pass"]), int(x["rows"])) for x in (pj.get("summary") or {}).values()
               if x.get("arm") in ("direct_answer", "fewshot_requested")}
        k_, n_ = sum(v[0] for v in req.values()), sum(v[1] for v in req.values())
        self.p(ln, m, 3, k_, "F7: boxed rows of the requested arms")
        self.p(ln, m, 4, n_, "F7: rows of the requested arms")
        self.p(ln, m, 5, Fraction(k_, n_) if n_ else None, "F7: share")
        got = {}
        o6 = m.start(6)
        for am in re.finditer(r"(zero-shot|five-shot) requested ((?:\d+ of \d+ on (?:" + TASK_RE[1:-1] + r")(?:, )?)+)",
                              m.group(6)):
            arm = "direct_answer" if am.group(1) == "zero-shot" else "fewshot_requested"
            for i in re.finditer(r"(\d+) of (\d+) on " + TASK_RE, am.group(2)):
                tk = TASK_ROW[i.group(3)]
                got[(tk, arm)] = True
                off = o6 + am.start(2)
                self.p(ln, i, 1, req.get((tk, arm), (None, None))[0], f"F7: {tk} {arm} boxed", off)
                self.p(ln, i, 2, req.get((tk, arm), (None, None))[1], f"F7: {tk} {arm} rows", off)
        self.T.text(f"{ln.where}: F7: arms and benchmarks listed", sorted(req), sorted(got))
        self.w(ln, m, 7, pj.get("run_key"), "F7 run key")

    def f1_inner(self, ln: Line) -> None:
        """F1's sentence on inner boxes: the restarts after a closed reasoning block, per ladder model,
        benchmark and forced arm, against the raw replies read here (ground truth); every model with one is named."""
        t, G = ln.text, getattr(self.F, "gt", None)
        m = re.search(r" A box that the model opens and closes inside the prefilled one is read as the reply's last "
                      r"complete box, whatever stands before it, for every model, as for the panel\.(?: One form of it "
                      r"is a restart, a new box begun after a closed reasoning block \(`[^`]*`\), read the same way: "
                      r"(.*?); no other ladder model writes one \(`([^`]+)`\)\.)?", t)
        audit = ((self.F.J.get("stop_counts") or {}).get("audit") or {})
        if not audit:          # no forced-stop audit on file (a synthetic set): the sentence is left out
            if m:
                self.fail(ln, "F1: a sentence on inner boxes with no forced-stop audit on file")
            return
        if not m:
            return self.fail(ln, "F1: the sentence on inner boxes is missing or not parsed")
        if not G:
            if m.group(1):
                ln.use(m.start(1), m.end(1))
            return
        words = {"zero-shot forced": "forced_direct", "five-shot forced": "fewshot_forced"}
        want = sorted((k, t_, a) for k in FILES if k in self.F.present for t_ in STOP_TASKS for a in GT.FORCED_ARMS
                      if G["unread"].get(k, {}).get(t_, {}).get("arms", {}).get(a, {}).get("restart"))
        got = []
        if m.group(1):
            self.w(ln, m, 2, STOP_AUDIT_JSON, "F1 inner boxes source")
            for i in re.finditer(NAME_RE + " " + VI + r" \(" + TASK_RE + r", (zero-shot forced|five-shot forced)\)",
                                 m.group(1)):
                k, tk, arm = KEY_OF[i.group(1)], TASK_ROW[i.group(3)], words[i.group(4)]
                got.append((k, tk, arm))
                self.p(ln, i, 2, G["unread"].get(k, {}).get(tk, {}).get("arms", {}).get(arm, {}).get("restart", 0),
                       f"F1 restarts {k} {tk} {arm} (raw replies, read here)", m.start(1))
        self.T.text(f"{ln.where}: F1 restarts, models named", want, sorted(got))

    def f1(self, ln: Line) -> None:
        t, F, root = ln.text, self.F, self.F.root
        m = re.search(r"Mistral Large 3 \(`([^`]+)`\): in " + V + r"% of five-shot forced replies it writes the option "
                      r"letter with a backslash, `[^`]+`, and this form accounts for " + V + " of the " + V + r" pp gap; "
                      r"in the zero-shot forced arm, " + V + " of the " + V + r" pp gap is the option's text written "
                      r"without its letter and " + V + r" a LaTeX text wrapper such as `[^`]+`\.", t)
        g = gap_json(root, MISTRAL_CHECK_JSON)
        if m:
            self.w(ln, m, 1, MISTRAL_CHECK_JSON, "F1 Mistral source")
            if g is None:
                self.fail(ln, f"F1: {MISTRAL_CHECK_JSON} is missing or holds no gap by form")
            else:
                gp, n = g["gap"], g["gap"]["n_arm"]
                frm = gp["by_form"]
                share = lambda c, arm: Fraction(100 * c, n[arm])  # noqa: E731
                self.p(ln, m, 2, g["forms_all"]["fewshot_forced"]["bs_letter"], "F1 Mistral share (\\C), five-shot")
                self.p(ln, m, 3, share(frm["letter with backslash (\\C)"]["fewshot_forced"], "fewshot_forced"),
                       "F1 Mistral (\\C) pp")
                self.p(ln, m, 4, share(gp["n"]["fewshot_forced"], "fewshot_forced"), "F1 Mistral five-shot gap")
                self.p(ln, m, 5, share(frm["option text, no letter"]["forced_direct"], "forced_direct"),
                       "F1 Mistral option-text pp")
                self.p(ln, m, 6, share(gp["n"]["forced_direct"], "forced_direct"), "F1 Mistral zero-shot gap")
                self.p(ln, m, 7, share(frm["LaTeX text wrapper"]["forced_direct"], "forced_direct"),
                       "F1 Mistral LaTeX wrapper pp")
            ml = F.x.get("mistral-large-2512", {}).get("cut", {}).get("bbh")
            if ml:
                self.T.printed(f"{ln.where}: F1 Mistral five-shot gap from the rows", m.group(4),
                               ml["gap_all_rows"]["fs_forced"])
                self.T.printed(f"{ln.where}: F1 Mistral zero-shot gap from the rows", m.group(6),
                               ml["gap_all_rows"]["forced"])
        elif "mistral-large-2512" in F.present and g is not None:
            self.fail(ln, "F1: the Mistral Large 3 clause is missing or not parsed")
        m = re.search(r"DeepSeek \(`([^`]+)`\) copies the prompt's `[^`]+` placeholder: zero-shot forced writes the "
                      r"letter in angle brackets \(`<B>`, " + V + " pp of the " + V + r" pp gap\) or the option's text \("
                      + V + r" pp\); five-shot forced writes `<answer>\(B\)` \(" + V + " pp of the " + V + r" pp gap\), "
                      r"the option's text \(" + V + r" pp\) or a LaTeX text wrapper \(" + V + r" pp\); " + VI + " of "
                      + VI + r" five-shot forced replies \(" + V + r"%\) hold no answer in their box: " + VI
                      + r" a bare `<answer>` placeholder and " + VI + r" empty \(([^)]*\)[^)]*|[^)]*)\); " + VI
                      + r" of them hold the right answer after the first `\}`, which the panel's stop removes\.", t)
        g = gap_json(root, NEBIUS_CHECK_JSON)
        if m:
            self.w(ln, m, 1, NEBIUS_CHECK_JSON, "F1 DeepSeek source")
            if g is None:
                self.fail(ln, f"F1: {NEBIUS_CHECK_JSON} is missing or holds no gap by form")
            else:
                gp, n = g["gap"], g["gap"]["n_arm"]
                frm = gp["by_form"]
                share = lambda c, arm: Fraction(100 * c, n[arm])  # noqa: E731
                ph = g["placeholders"]["bbh|fewshot_forced"]
                for gi, v, what in (
                        (2, share(frm["letter in angle brackets (<B>)"]["forced_direct"], "forced_direct"), "<B> pp"),
                        (3, share(gp["n"]["forced_direct"], "forced_direct"), "zero-shot gap"),
                        (4, share(frm["option text, no letter"]["forced_direct"], "forced_direct"),
                         "zero-shot option-text pp"),
                        (5, share(frm["answer tag around a letter (<answer>(B))"]["fewshot_forced"], "fewshot_forced"),
                         "<answer>(B) pp"),
                        (6, share(gp["n"]["fewshot_forced"], "fewshot_forced"), "five-shot gap"),
                        (7, share(frm["option text, no letter"]["fewshot_forced"], "fewshot_forced"),
                         "five-shot option-text pp"),
                        (8, share(frm["LaTeX text wrapper"]["fewshot_forced"], "fewshot_forced"),
                         "five-shot LaTeX wrapper pp"),
                        (9, ph["rows"], "placeholder rows"), (10, ph["n_arm"], "five-shot rows"),
                        (11, Fraction(100 * ph["rows"], ph["n_arm"]), "placeholder share"),
                        (12, (ph.get("by_kind") or {}).get("<answer>", 0), "bare <answer> rows"),
                        (13, sum(v for k_, v in (ph.get("by_kind") or {}).items() if k_ != "<answer>"), "empty rows"),
                        (15, ph.get("content_correct_without_cut"), "right answer after the first brace")):
                    self.p(ln, m, gi, v, f"F1 DeepSeek {what} from {NEBIUS_CHECK_JSON}")
                kinds = {"empty box": "empty box", "empty boxes": "empty box", "empty `()`": "empty parentheses ()",
                         "empty reply": "empty reply", "empty replies": "empty reply"}
                got = {kinds.get(e.group(2)): e for e in re.finditer(VI + r" (empty boxes|empty box|empty `\(\)`|"
                                                                    r"empty replies|empty reply)", m.group(14))}
                want_k = {k_: v for k_, v in (ph.get("by_kind") or {}).items() if k_ != "<answer>"}
                self.T.text(f"{ln.where}: F1 DeepSeek empty kinds", sorted(want_k), sorted(k_ for k_ in got if k_))
                for k_, e in got.items():
                    if k_ in want_k:
                        self.p(ln, e, 1, want_k[k_], f"F1 DeepSeek {k_} rows", m.start(14))
            dx = F.x.get(DS, {}).get("cut", {}).get("bbh")
            if dx:
                self.T.printed(f"{ln.where}: F1 DeepSeek zero-shot gap from the rows", m.group(3),
                               dx["gap_all_rows"]["forced"])
                self.T.printed(f"{ln.where}: F1 DeepSeek five-shot gap from the rows", m.group(6),
                               dx["gap_all_rows"]["fs_forced"])
                self.T.printed(f"{ln.where}: F1 DeepSeek five-shot rows from the rows", m.group(10),
                               dx["rows_by_arm"]["fewshot_forced"])
        elif DS in F.present and g is not None:
            self.fail(ln, "F1: the DeepSeek clause is missing or not parsed")
        self.f1_inner(ln)
        self.f1_stop(ln)
        m = re.search(r"GSM8K, MATH-500 and HotpotQA, computed here: the largest content − strict gap over all rows of a "
                      r"forced arm \(zero-shot or five-shot\) is (.*?)\.(?= The cluster| [A-Z]|$)", t)
        if m:
            items = list(re.finditer(V + " pp for " + NAME_RE + r" \(" + TASK_RE + r", (zero-shot forced|five-shot "
                                     r"forced)\)", m.group(1)))
            self.T.text(f"{ln.where}: F1 other-benchmark models", list(F.other_max),
                        [KEY_OF[i.group(2)] for i in items])
            for i in items:
                k = KEY_OF[i.group(2)]
                v, tk, arm = F.other_max[k]
                self.p(ln, i, 1, v, f"F1 {k} largest other-benchmark forced gap", m.start(1))
                self.T.text(f"{ln.where}: F1 {k} where", (TASK_NAME[tk], {"forced": "zero-shot forced",
                                                                          "fs_forced": "five-shot forced"}[arm]),
                            (i.group(3), i.group(4)))
        elif F.other_max:
            self.fail(ln, "F1: the other-benchmark sentence is missing or not parsed")
        m = re.search(r"The cluster rungs, same classes \(`[^`]+`, from `[^`]+`; pp of all rows of the arm; the three "
                      r"largest forms of " + V + r" pp or more\): (.*)\.$", t)
        fj = root / FORMS_JSON
        if m:
            self.p(ln, m, 1, FORM_MIN, "F1 form threshold")
            d = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {"models": {}}
            want_models = [k for k in d.get("models", {}) if k in F.present]
            mods = list(re.finditer(NAME_RE + r": (.*?)(?=; " + NAME_RE + ": |$)", m.group(2)))
            self.T.text(f"{ln.where}: F1 cluster models", want_models, [KEY_OF[x.group(1)] for x in mods])
            for mm in mods:
                k = KEY_OF[mm.group(1)]
                off = m.start(2) + mm.start(2)
                arms = list(re.finditer(r"(five-shot forced|zero-shot forced) " + V + r" pp \[([^\]]*)\]", mm.group(2)))
                jarms = d["models"].get(k, {}).get("arms", {})
                self.T.text(f"{ln.where}: F1 {k} arms", [("five-shot forced" if a == "fewshot_forced" else
                                                          "zero-shot forced") for a in jarms],
                            [a.group(1) for a in arms])
                for a in arms:
                    arm = "fewshot_forced" if a.group(1).startswith("five") else "forced_direct"
                    ja = jarms.get(arm, {})
                    self.p(ln, a, 2, ja.get("gap_pp"), f"F1 {k} {arm} gap as in {FORMS_JSON}", off)
                    x = F.x.get(k, {}).get("cut", {}).get("bbh")
                    if x:
                        self.T.printed(f"{ln.where}: F1 {k} {arm} gap from the rows", a.group(2),
                                       x["gap_all_rows"][LABEL[arm]])
                    forms = sorted([f for f in ja.get("forms", []) if f["pp"] >= FORM_MIN], key=lambda f: -f["pp"])[:3]
                    got = [re.match(r"^(.*) (-?\d+\.\d+)$", s) for s in a.group(3).split("; ")]
                    self.T.text(f"{ln.where}: F1 {k} {arm} forms", [f["form"] for f in forms],
                                [x_.group(1) if x_ else None for x_ in got])
                    for f, x_ in zip(forms, got):
                        if x_:
                            self.T.printed(f"{ln.where}: F1 {k} {arm} {f['form']}", x_.group(2), f["pp"])
                    ln.use(off + a.start(3), off + a.end(3))
        elif fj.exists() and any(k in F.present for k in json.loads(fj.read_text(encoding="utf-8")).get("models", {})):
            self.fail(ln, "F1: the cluster-rung clause is missing or not parsed")

    def f1_stop(self, ln: Line) -> None:
        """F1's clause on the stop's loss: the unread and unfinished counts per ladder model against the raw replies
        read here (ground truth); the API models' numbers against the source the clause cites (FULL_REPLIES)."""
        t, F = ln.text, self.F
        G = getattr(F, "gt", None)
        m = re.search(r" The stop's loss: replies the forced arms' stop at the first `\}` left with the box open that "
                      r"the readers do not read \(([^;]+); `([^`]+)`\), zero-shot / five-shot forced: (.*?)\. Of "
                      r"these, the text stops on a connective or holds two options \([^;]*; the rest stop inside "
                      r"another group: [^)]*\), zero-shot / five-shot forced: (.*?)\. Time ranges the stop cut at "
                      r"\"to\" \([^)]*\) are not in the loss: [^:]*?, zero-shot / five-shot forced: (.*?)\."
                      r"(?: Against the API models' full "
                      r"replies \(`([^`]+)`\), the unread replies that are correct in the box the model closed: (.*?); "
                      r"replies read as correct where the full box is not: (.*?)\.)?(?= GSM8K, MATH-500| The cluster|$)",
                      t)
        if not m:
            if G and (F.J.get("stop_counts") or {}).get("audit"):
                self.fail(ln, "F1: the stop's-loss clause is missing or not parsed")
            return
        if not G:
            return self.fail(ln, "F1: the stop's-loss clause is printed but the raw replies are not here to check it")
        self.w(ln, m, 1, G["rule"].version, "F1 stop rule version")
        self.w(ln, m, 2, STOP_AUDIT_JSON, "F1 stop source")
        # the clause's examples are fixed texts; its count of temporal_sequences items whose options all
        # begin at different times is recounted from the items read here (the options' text before " to "); the last
        # example is a fixed reference label the text prints, not a count
        for ex in ('"10 \\text{ blue and }", "({A, D}"', '"8pm \\text{ to }"', "(see the completion rule's open_range class)"):
            j = t.find(ex)
            self.T.text(f"{ln.where}: F1 stop example", ex, t[j:j + len(ex)] if j >= 0 else None)
            if j >= 0:
                ln.use(j, j + len(ex))
        mm = re.search(r"the (\d+) temporal_sequences items' options begin at different times", t)
        if mm:
            items = [o for _g, _a, task, o in G["reader"].bbh.values() if task == "temporal_sequences"]
            distinct = sum(1 for o in items if len({x.split(" to ")[0].strip() for x in o}) == len(o) > 1)
            self.p(ln, mm, 1, distinct, "F1 temporal_sequences items whose options begin at different times")
            self.T.num(f"{ln.where}: F1 every temporal_sequences item has options that begin at different times",
                       distinct, len(items))
        else:
            self.fail(ln, "F1: the count of temporal_sequences items is missing")
        cell = NAME_RE + r": GSM8K " + VI + " / " + VI + r", BIG-Bench Hard " + VI + " / " + VI
        for grp, field in ((3, "unread"), (4, "unfinished"), (5, "range_read")):
            items = list(re.finditer(cell, m.group(grp)))
            want = [k for k in FILES if k in F.present]
            if field != "unread":
                want = [k for k in want if any(G["unread"].get(k, {}).get(t_, {}).get("arms", {}).get(a, {})
                                               .get(field) for t_ in STOP_TASKS for a in GT.FORCED_ARMS)]
            self.T.text(f"{ln.where}: F1 stop {field} models", want, [KEY_OF[i.group(1)] for i in items])
            for i in items:
                k = KEY_OF[i.group(1)]
                for gi, (t_, arm) in zip((2, 3, 4, 5), [(t_, a) for t_ in STOP_TASKS for a in GT.FORCED_ARMS]):
                    v = G["unread"].get(k, {}).get(t_, {}).get("arms", {}).get(arm, {}).get(field, 0)
                    self.p(ln, i, gi, v, f"F1 stop {field} {k} {t_} {arm} (raw replies, read here)", m.start(grp))
        if m.group(6) is None:
            return
        self.w(ln, m, 6, FULL_REPLIES_JSON, "F1 full-replies source")
        p = F.root / FULL_REPLIES_JSON
        if not p.exists():
            return self.fail(ln, f"F1: {FULL_REPLIES_JSON} is cited but absent")
        fr = json.loads(p.read_text(encoding="utf-8"))
        agg = {}
        for c in fr["cells"]:
            if c["version"] != G["rule"].version:
                continue
            d = agg.setdefault(c["model"], Counter())
            d["compared"] += c.get("compared", 0)
            d["false_gain"] += c.get("false_gain", 0)
            # the replies the pipeline leaves unread, as the script counts them per cell
            d["unread_compared"] += c.get("unread_compared", 0)
            d["unread_false_loss"] += c.get("unread_false_loss", 0)
        for grp, a_f, b_f in ((7, "unread_false_loss", "unread_compared"), (8, "false_gain", None)):
            for i in re.finditer(NAME_RE + " " + VI + " of " + VI, m.group(grp)):
                k = KEY_OF[i.group(1)]
                d = agg.get(k, Counter())
                self.p(ln, i, 2, d[a_f], f"F1 full replies {k} {a_f}", m.start(grp))
                self.p(ln, i, 3, d[b_f] if b_f else d["compared"] - d["unread_compared"],
                       f"F1 full replies {k} {b_f or 'read'}", m.start(grp))

    def f2(self, ln: Line) -> None:
        t, root = ln.text, self.F.root
        m = re.search(r"two replies to the same prompt differed in (\d+) of (\d+) pilot-against-check pairs, and in "
                      r"(\d+) of (\d+) pairs in each of three separate tests: at temperature (\d+), with `top_k` (\d+), "
                      r"and with seed (\d+)\. Scored strict, those (\d+) same-cap pairs gave the same answer in all "
                      r"(\d+)\. On MATH-500, all (\d+) pilot replies rerun at the raised cap \((\d+) requested, (\d+) "
                      r"chain of thought\) differ, in the sense that the rerun does not begin with the pilot's cut "
                      r"reply\.", t)
        if not m:
            if DS in self.F.present and (root / NEBIUS_CHECK).exists():
                self.fail(ln, "F2 not parsed")
            return
        src = (root / NEBIUS_CHECK).read_text(encoding="utf-8")
        rows = {r[0]: r for _, h, rs in md_tables(src) if h[:2] == ["Benchmark, arm", "Pairs"] for r in rs}

        def cell(label, col):
            r = rows.get(label)
            return None if r is None else int(r[col].replace(",", ""))
        pc = "GSM8K chain of thought: pilot vs determinism check"
        temp = next((k for k in rows if re.search(r"two calls at temperature (\d+)$", k)), None)
        topk = next((k for k in rows if re.search(r"two calls with `top_k` (\d+)$", k)), None)
        seed = next((k for k in rows if re.search(r"two calls with seed (\d+)$", k)), None)
        mreq = next((k for k in rows if k.startswith("MATH-500 requested")), None)
        mcot = next((k for k in rows if k.startswith("MATH-500 chain of thought")), None)
        self.p(ln, m, 1, cell(pc, 4), "F2 pilot-vs-check pairs differing")
        self.p(ln, m, 2, cell(pc, 1), "F2 pilot-vs-check pairs")
        three = [x for x in (temp, topk, seed) if x]
        same = len(three) == 3 and len({(cell(x, 1), cell(x, 4)) for x in three}) == 1
        self.T.text(f"{ln.where}: F2 the three tests have the same counts", True, same)
        self.p(ln, m, 3, cell(temp, 4) if temp else None, "F2 temperature-0 pairs differing")
        self.p(ln, m, 4, cell(temp, 1) if temp else None, "F2 temperature-0 pairs")
        for gi, lab in ((5, temp), (6, topk), (7, seed)):
            self.w(ln, m, gi, re.search(r"(\d+)$", lab).group(1) if lab else None, f"F2 setting {gi}")
        s35 = re.search(r"Scored strict, the (\d+) GSM8K same-cap pairs give the same answer in all (\d+) cases", src)
        n_same = sum(cell(x, 1) or 0 for x in [pc] + three)
        self.p(ln, m, 8, n_same, "F2 same-cap pairs (sum of the table)")
        self.w(ln, m, 8, s35.group(1) if s35 else None, "F2 same-cap pairs as in the source")
        self.w(ln, m, 9, s35.group(2) if s35 else None, "F2 same strict answer as in the source")
        self.p(ln, m, 10, (cell(mreq, 4) or 0) + (cell(mcot, 4) or 0), "F2 reruns differing")
        self.p(ln, m, 11, cell(mreq, 4) if mreq else None, "F2 MATH-500 requested reruns differing")
        self.p(ln, m, 12, cell(mcot, 4) if mcot else None, "F2 MATH-500 chain-of-thought reruns differing")
        self.T.text(f"{ln.where}: F2 every rerun differs", True,
                    bool(mreq and mcot and cell(mreq, 4) == cell(mreq, 1) and cell(mcot, 4) == cell(mcot, 1)))

    # -- generic constants (any line) ------------------------------------------------------------------------------
    def constants(self, ln: Line) -> None:
        for m in re.finditer(r"\b(\d+)(?= draws\b|-draw\b)", ln.text):
            self.p(ln, m, 1, self.F.n_all, "draws")
        for m in re.finditer(r"\b(\d+)(?= discovery draws\b)", ln.text):
            self.p(ln, m, 1, self.F.n_disc, "discovery draws")
        for rx, v in ((r"a step of (0\.1) pp or more upwards", FLAT["pp"]), (r"(0\.1) pp or more downwards", FLAT["pp"]),
                      (r"\(for w, a share: (0\.001)\)", FLAT["w"])):
            for m in re.finditer(rx, ln.text):
                self.p(ln, m, 1, v, "direction threshold")
        for m in re.finditer(r"\b(?:within|tolerance) (1e-0?9)\b", ln.text):
            self.T.num(f"{ln.where}: tolerance", float(m.group(1)), TOL)
            ln.use(*m.span(1))
        for m in re.finditer(r"every row of the (\d+) configurations", ln.text):
            self.p(ln, m, 1, self.F.hotpot_configs, "HotpotQA configurations")
        for m in re.finditer(r"\((\d+) tokens; (\d+) on HotpotQA\)", ln.text):
            self.p(ln, m, 1, forced_cap("gsm8k", self.F.root), "forced cap")
            self.p(ln, m, 2, forced_cap("hotpot", self.F.root), "forced cap, HotpotQA")

    # -- note ----------------------------------------------------------------------------------------------------
    def what_is_in(self, ln: Line) -> None:
        m = re.match(r"^" + NAME_RE + r": (complete|partial|missing); (.*?)\. Ladder: (.*?)\. Size: (.*?)\. Precision: "
                     r"(.*)\.$", ln.text)
        if not m:
            if ln.text.startswith("Sizes with their sources"):
                return self.size_order_note(ln)
            return self.fail(ln, "what-is-in line not parsed")
        k = KEY_OF[m.group(1)]
        st = "missing" if k in self.F.missing else ("partial" if k in self.F.partial else "complete")
        self.w(ln, m, 2, st, f"{k} status")
        self.state(ln, k, m.group(3), m.start(3))
        self.w(ln, m, 4, FAMILY_OF.get(k, OUTSIDE), f"{k} ladder")
        self.size_cell(ln, m, 5, k, with_source=False)

    def size_order_note(self, ln: Line) -> None:
        want = "Rungs are ordered by total parameters" if self.ordered() else "Rungs are not ordered by total parameters"
        self.T.text(f"{ln.where}: order of the rungs", True, want in ln.text)

    def ordered(self) -> bool:
        for name, f in FAMILIES.items():
            tot = [self.F.sizes.get(k, {}).get("total") for k, _ in f["rungs"]]
            if None in tot or not all(Fraction(b) > Fraction(a) for a, b in zip(tot, tot[1:])):
                return False
        return True

    def size_cell(self, ln: Line, m, grp: int, key: str, with_source: bool = True, off: int = 0) -> None:
        """A printed size against the recorded one: the total, the active parameters and the architecture, each as the
        source writes them, and the source named."""
        txt = m.group(grp)
        a = m.start(grp) + off
        z = self.F.sizes.get(key, {})
        src = r"(?: \((.*)\))?$"
        if z.get("published"):
            # published figures; DeepSeek's active parameters for input and output apart
            mm = re.match(r"^(\d+(?:\.\d+)?)B total, (\d+(?:\.\d+)?)B active for input and (\d+(?:\.\d+)?)B for "
                          r"output, (mixture of experts)" + src, txt)
            fields = ("total", "active_input", "active_output")
            if not mm:
                mm = re.match(r"^(\d+(?:\.\d+)?)B total, (\d+(?:\.\d+)?)B active, (mixture of experts)" + src, txt)
                fields = ("total", "active")
            if not mm:
                return self.fail(ln, f"{key}: size not parsed: {txt!r}")
            self.T.text(f"{ln.where}: {key} size form", "input/output" if z.get("active_input") else "one active",
                        "input/output" if "active_input" in fields else "one active")
            for i, f in enumerate(fields, 1):
                self.w(ln, mm, i, z.get(f), f"{key} {f} parameters (published)", a)
            last = len(fields) + 1
            self.w(ln, mm, last, z.get("arch"), f"{key} architecture (published)", a)
            if mm.group(last + 1) is not None:
                self.w(ln, mm, last + 1, published_source(z) if with_source else z.get("caveat"),
                       f"{key} size source" if with_source else f"{key} caveat of the published size", a)
            elif with_source or z.get("caveat"):
                self.fail(ln, f"{key}: size printed without its " + ("source" if with_source else "caveat"))
            return
        forms = ((r"^(\d+(?:\.\d+)?)B, dense" + src, "inventory dense"),
                 (r"^(\d+(?:\.\d+)?)B by name, dense" + src, "name dense"),
                 (r"^(\d+(?:\.\d+)?)B total, (\d+(?:\.\d+)?)B active, mixture of experts" + src, "inventory moe"),
                 (r"^(\d+(?:\.\d+)?)B total by name; active parameters and architecture not recorded" + src, "name"))
        for rx, kind in forms:
            mm = re.match(rx, txt)
            if mm:
                break
        else:
            return self.fail(ln, f"{key}: size not parsed: {txt!r}")
        want_kind = ("name dense" if z.get("by_name") and z.get("arch") == "dense" else
                     "name" if z.get("by_name") and not z.get("arch") else
                     "inventory dense" if z.get("arch") == "dense" else "inventory moe" if z.get("arch") else None)
        self.T.text(f"{ln.where}: {key} size form", want_kind, kind)
        self.w(ln, mm, 1, z.get("total"), f"{key} total parameters", a)
        last = 2
        if kind == "inventory moe":
            self.w(ln, mm, 2, z.get("active"), f"{key} active parameters", a)
            last = 3
        if mm.group(last) is not None:
            ln.use(a + mm.start(last), a + mm.end(last))
            if z.get("table"):
                self.T.text(f"{ln.where}: {key} size source", f"`manifests/dataset2_inventory.md`, {z['table']}",
                            mm.group(last))
            elif z.get("by_name"):
                self.T.text(f"{ln.where}: {key} size source names the name", True, "name" in mm.group(last))
                if z.get("arch"):
                    self.T.text(f"{ln.where}: {key} architecture source", True, "config.json" in mm.group(last))
        elif with_source:
            self.fail(ln, f"{key}: size printed without its source")

    def what_waits(self, ln: Line) -> None:
        m = re.match(r"^" + NAME_RE + r": (.*?)\. Its family ladder \(([^)]*)\)", ln.text)
        if m:
            k = KEY_OF[m.group(1)]
            self.T.text(f"{ln.where}: {k} waits", True, k in self.F.missing or k in self.F.partial)
            self.state(ln, k, m.group(2), m.start(2))
            self.w(ln, m, 3, FAMILY_OF.get(k, "none"), f"{k} family")

    def precision_phrase(self) -> str:
        known = {k: v for k, v in self.F.prec.items() if v}
        if not known:
            return "not recorded here"
        cnt: dict = {}
        for v in known.values():
            cnt[v] = cnt.get(v, 0) + 1
        main = max(sorted(cnt), key=lambda x: cnt[x])
        other: dict = {}
        for k, v in sorted(known.items()):
            if v != main:
                other.setdefault(v, []).append(k)
        miss = sorted(k for k, v in self.F.prec.items() if not v)
        return (main + ("" if not other else ", except " + "; ".join(" and ".join(f"`{k}`" for k in ks) + f" in {v}"
                                                                     for v, ks in sorted(other.items())))
                + ("" if not miss else "; not recorded for " + ", ".join(f"`{k}`" for k in miss)))

    def limits(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.search(r"where each of the (\d+) ladder models falls among the (\d+)(?: or (\d+))? panel models", t)
        if m:
            self.p(ln, m, 1, len(F.present), "models in")
            ns = sorted({int(F.panel[tk]["cot_minus_forced_strict"].notna().sum()) for tk in TASKS})
            self.p(ln, m, 2, ns[0], "panel size (smallest)")
            if m.group(3):
                self.p(ln, m, 3, ns[-1], "panel size (largest)")
            self.T.text(f"{ln.where}: panel sizes", len(ns), 2 if m.group(3) else 1)
            return
        m = re.match(r"^" + FAM_RE + r": (\d+) of (\d+) rungs(?: \((\d+) from the panel, (\d+) from the ladder\))?", t)
        if m:
            f = F.fam[m.group(1)]
            self.p(ln, m, 2, len(f["present"]), "rungs in")
            self.p(ln, m, 3, len(f["rungs"]), "rungs")
            if m.group(4):
                self.p(ln, m, 4, sum(r["src"] == "panel" for r in f["rungs"]), "panel rungs")
                self.p(ln, m, 5, sum(r["src"] == "ladder" and r["present"] for r in f["rungs"]), "ladder rungs in")
            return
        m = re.search(r"against the panel's (.*)\), kernels and serving route", t)
        if m:
            self.w(ln, m, 1, self.precision_phrase(), "the panel's precision")
            return
        if t.startswith("Rungs are"):
            return self.size_order(ln)
        m = re.search(r"the ladder adds (\d+) single models to (\d+) families of (\d+) or (\d+) rungs(?:, and (\d+) "
                      r"outside them \(([^)]*)\))?, which", t)
        if m:
            infam = [k for k in F.present if k in FAMILY_OF]
            out = [k for k in F.present if k not in FAMILY_OF]
            sizes = sorted({len(f["rungs"]) for f in FAMILIES.values()})
            self.p(ln, m, 1, len(infam), "ladder models in the families")
            self.p(ln, m, 2, len(FAMILIES), "families")
            self.p(ln, m, 3, sizes[0], "rungs (fewest)")
            self.p(ln, m, 4, sizes[-1], "rungs (most)")
            if m.group(5):
                self.p(ln, m, 5, len(out), "ladder models outside the families")
                self.w(ln, m, 6, ", ".join(NAMES[k][1] for k in out), "the models outside")
            return

    def size_order(self, ln: Line) -> None:
        """'Rungs are ordered by total parameters ...': per family, every rung's recorded total and active parameters,
        and whether the active parameters rise with the total."""
        t, F = ln.text, self.F
        head = "Rungs are ordered by total parameters" if self.ordered() else None
        m0 = re.match(r"^(Rungs are (?:not )?ordered by total parameters)(?: in [^(]*)? \(sizes and their sources in "
                      r"section 4\)\. ", t)
        if not m0:
            return self.fail(ln, "size-order line not parsed")
        if head:
            self.T.text(f"{ln.where}: order of the rungs", head, m0.group(1))
        names = "|".join(re.escape(f) for f in FAMILIES)
        for fm in re.finditer(r"(" + names + r"): (.*?); (the active parameters .*?)(?=\. (?:" + names + r"): |\.$)", t):
            fam = fm.group(1)
            rungs = [k for k, _ in FAMILIES[fam]["rungs"]]
            steps = fm.group(2).split(" → ")
            if len(steps) != len(rungs):
                self.fail(ln, f"{fam}: {len(steps)} sizes for {len(rungs)} rungs")
                continue
            pos = fm.start(2)
            act = []
            for k, st in zip(rungs, steps):
                z = F.sizes.get(k, {})
                sm = re.match(r"^(\d+(?:\.\d+)?)B (dense|total)(?:, (\d+(?:\.\d+)?)B active)?$", st)
                pos = t.index(st, pos)
                if not sm:
                    self.fail(ln, f"{fam} {k}: size step not parsed: {st!r}")
                    continue
                self.w(ln, sm, 1, z.get("total"), f"{fam} {k} total", pos)
                if sm.group(2) == "dense":
                    self.T.text(f"{ln.where}: {fam} {k} dense", "dense", z.get("arch"))
                elif sm.group(3):
                    self.w(ln, sm, 3, z.get("active"), f"{fam} {k} active", pos)
                else:
                    self.T.text(f"{ln.where}: {fam} {k} active not recorded", None, z.get("active"))
                act.append(z.get("active"))
                pos += len(st)
            if None in act:
                want = "the active parameters are not recorded for " + ", ".join(
                    SIZE.get(k, k) for k, a in zip(rungs, act) if a is None)
            elif all(Fraction(b) > Fraction(a) for a, b in zip(act, act[1:])):
                want = "the active parameters rise with the total"
            else:
                want = ("the active parameters (" + " → ".join(f"{a}B" for a in act) + ") do not rise with the total, "
                        "so a direction over size there is not one over active parameters")
            self.w(ln, fm, 3, want, f"{fam}: active parameters over the rungs")

    # -- every section ------------------------------------------------------------------------------------------
    def f8(self, ln: Line) -> None:
        """F8: the cut checked against a rerun only under batch-invariant kernels; the default-kernel models and the
        shares of their outputs the cut shortens."""
        t, F = ln.text, self.F
        dk = [k for k in F.present if F.kernels.get(k, "").startswith("default")]
        dkf = {e["model_key"]: e for e in F.equiv if e.get("kernels") == "default"}
        if dk and all(k in dkf for k in dk):
            # the default-kernel check on file, its rates and its caveat
            m = re.search(r"\. Under batch-invariant kernels: (.*?)\. (" + NAME_NC + r"(?:, " + NAME_NC
                          + r")*) ran with default kernels, under which a rerun need not repeat its tokens: (.*?)\. "
                          r"(These rows were chosen .*?)\. The outputs the cut shortens, share of the arm's rows: (.*)\.$",
                          t)
            if not m:
                return self.fail(ln, "F8 not parsed (default-kernel check on file)")
            self.equiv_list(ln, m, 1)
            self.w(ln, m, 2, ", ".join(NAMES[k][1] for k in dk), "F8: default-kernel models")
            self.dk_equiv_list(ln, m, 3)
            self.w(ln, m, 4, dk_caveat(dkf), "F8: the default-kernel check's caveat")
            g = 5
        else:
            m = re.search(r"was checked only under batch-invariant kernels: (.*?)\. (" + NAME_NC + r"(?:, " + NAME_NC
                          + r")*) ran with default kernels, "
                          r"under which a rerun need not repeat its tokens, and was not checked, so the agreement of its "
                          r"cut copy with a run at the panel's caps is untested\. The outputs the cut shortens, share of "
                          r"the arm's rows: (.*)\.$", t)
            if not m:
                return self.fail(ln, "F8 not parsed")
            self.equiv_list(ln, m, 1)
            self.w(ln, m, 2, ", ".join(NAMES[k][1] for k in dk), "F8: default-kernel models")
            g = 3
        for mm in re.finditer(NAME_RE + r": (.*?); no other raised arm has an output cut", m.group(g)):
            k = KEY_OF[mm.group(1)]
            off = m.start(g) + mm.start(2)
            want = [(tk, a) for tk in TASKS if tk in F.x[k]["cut"] for a in ("requested", "fs_requested", "cot")
                    if F.x[k]["cut"][tk].get("cut_at_panel_cap", {}).get(a)]
            items = list(re.finditer(TASK_RE + r" (requested|cot|fs_requested) " + V + r"% \(" + VI + " of " + VI + r"\)",
                                     mm.group(2)))
            self.T.text(f"{ln.where}: F8 {k} arms cut", sorted(want), sorted((TASK_ROW[i.group(1)], i.group(2))
                                                                            for i in items))
            for i in items:
                x = F.x[k]["cut"][TASK_ROW[i.group(1)]]
                raw = {"requested": "direct_answer", "cot": "zero_shot_cot", "fs_requested": "fewshot_requested"}
                n, c = x["rows_by_arm"].get(raw[i.group(2)]), x["cut_at_panel_cap"].get(i.group(2))
                self.p(ln, i, 3, None if not n else Fraction(100 * c, n), f"F8 {k} share cut", off)
                if c and Fraction(i.group(3).replace(",", "")) == 0:
                    self.fail(ln, f"F8 {k}: a nonzero count ({c}) printed as a zero share")
                self.p(ln, i, 4, c, f"F8 {k} cut", off)
                self.p(ln, i, 5, n, f"F8 {k} rows", off)

    def equiv_list(self, ln: Line, m, grp: int) -> None:
        """The cut-equivalence checks on file, as F8 and the cut-equivalence item print them."""
        F = self.F
        items = list(re.finditer(r"(?:" + NAME_NC + r"|[\w.-]+)(?: \((?:batch-invariant|default \(-dk fallback\)) "
                                 r"kernels\))?: (\d+) of (\d+) rerun rows identical to the pilot's cut copy,? \(?(\d+) of "
                                 r"(\d+) cut ones\)?", m.group(grp)))
        bi = [e for e in F.equiv if e.get("kernels") != "default"]
        self.T.text(f"{ln.where}: equivalence checks listed", len(bi), len(items))
        for i, e in zip(items, bi):
            off = m.start(grp)
            self.p(ln, i, 1, e.get("identical"), "equivalence: identical rows", off)
            self.p(ln, i, 2, e.get("rows"), "equivalence: rows", off)
            self.p(ln, i, 3, e.get("identical_among_cut"), "equivalence: identical cut rows", off)
            self.p(ln, i, 4, e.get("cut_in_pilot"), "equivalence: cut rows", off)

    def dk_equiv_list(self, ln: Line, m, grp: int) -> None:
        """The default-kernel cut-equivalence checks on file, as F8 and the cut-equivalence item print them: identical
        rows, the content verdict's agreement and its changes, and the file."""
        F = self.F
        dkf = [e for e in F.equiv if e.get("kernels") == "default"]
        items = list(re.finditer(NAME_RE + r" \(default kernels\): (\d+) of (\d+) rerun rows identical to the cut copy, "
                                 r"content verdict the same on (\d+) of (\d+) \((\d+) wrong to right, (\d+) right to "
                                 r"wrong\), on rows the run cut at the panel's cap \(`([^`]+)`\)", m.group(grp)))
        self.T.text(f"{ln.where}: default-kernel equivalence checks listed", len(dkf), len(items))
        for i, e in zip(items, dkf):
            off = m.start(grp)
            s = e.get("scored") or {}
            f = s.get("content_flips") or {}
            self.w(ln, i, 1, NAMES[e["model_key"]][1] if e["model_key"] in NAMES else e["model_key"],
                   "default-kernel equivalence: model", off)
            self.p(ln, i, 2, e.get("identical"), "default-kernel equivalence: identical rows", off)
            self.p(ln, i, 3, e.get("rows"), "default-kernel equivalence: rows", off)
            self.p(ln, i, 4, s.get("content_verdict_same"), "default-kernel equivalence: content verdict same", off)
            self.p(ln, i, 5, s.get("rows"), "default-kernel equivalence: scored rows", off)
            self.p(ln, i, 6, f.get("wrong_to_right"), "default-kernel equivalence: wrong to right", off)
            self.p(ln, i, 7, f.get("right_to_wrong"), "default-kernel equivalence: right to wrong", off)
            self.w(ln, i, 8, e["_file"], "default-kernel equivalence: file", off)

    def preamble(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.match(r"^# Size ladder: analysis \(" + STAMP + r"; (\d+) of (\d+) models in\)$", t)
        if m:
            self.w(ln, m, 1, F.J.get("created"), "created")
            self.p(ln, m, 2, len(F.present), "models in")
            self.p(ln, m, 3, len(FILES), "ladder models")
            return
        m = re.search(r"\((\d{1,3}(?:,\d{3})*) requested; (\d{1,3}(?:,\d{3})*) chain of thought, (\d{1,3}(?:,\d{3})*) on "
                      r"HotpotQA\)\. Supplementary: the uncut outputs at the raised cap \((\d{1,3}(?:,\d{3})*)\)", t)
        if m:
            caps = {tk: yaml.safe_load((F.root / "config" / f"decoding_ladder2_{tk}.yaml").read_text(encoding="utf-8"))
                    for tk in TASKS}
            self.p(ln, m, 1, caps["gsm8k"]["cut_caps"]["direct_answer"], "panel cap, requested")
            self.p(ln, m, 2, caps["gsm8k"]["cut_caps"]["zero_shot_cot"], "panel cap, chain of thought")
            self.p(ln, m, 3, caps["hotpot"]["cut_caps"]["zero_shot_cot"], "panel cap, HotpotQA chain of thought")
            self.p(ln, m, 4, caps["gsm8k"]["max_new_tokens"]["zero_shot_cot"], "raised cap")
            same = all(caps[tk]["cut_caps"]["direct_answer"] == caps["gsm8k"]["cut_caps"]["direct_answer"]
                       and caps[tk]["max_new_tokens"]["zero_shot_cot"] == caps["gsm8k"]["max_new_tokens"]["zero_shot_cot"]
                       for tk in TASKS) and all(caps[tk]["cut_caps"]["zero_shot_cot"] == caps["gsm8k"]["cut_caps"]
                                                ["zero_shot_cot"] for tk in DISC_TASKS)
            self.T.text(f"{ln.where}: the caps are the same on every benchmark it names", True, same)
            return
        for m in re.finditer(r"(\d+\.\d+) as (\d+\.\d+)", t):
            self.w(ln, m, 2, format(Fraction(m.group(1)), f".{len(m.group(2).split('.')[1])}f"),
                   f"rounding example {m.group(1)}")
            ln.use(*m.span(1))

    def asks(self, ln: Line) -> None:
        """Section 3, line by line."""
        t, F = ln.text, self.F
        for m in re.finditer(FAM_RE + r" (\d+) of (\d+) rungs", t):
            f = F.fam[m.group(1)]
            self.p(ln, m, 2, len(f["present"]), f"{m.group(1)} rungs in")
            self.p(ln, m, 3, len(f["rungs"]), f"{m.group(1)} rungs")
        if t.startswith("Goal: within-family"):
            m = re.search(r"\(Llama-3\.1-70B: (.*?)\)\. The (\d+) models in ran", t)
            if not m:
                return self.fail(ln, "goal line not parsed")
            self.state(ln, "llama-3.1-70b-instruct-fp8", m.group(1), m.start(1))
            self.p(ln, m, 2, len(F.present), "models in")
        elif t.startswith("Model lists:"):
            sp = F.spend
            m = re.search(r"Mistral Large 3 \$(\d+\.\d+) for the run, \$(\d+\.\d+) with the forced test", t)
            if m or sp.get("mistral"):
                if not m or not sp.get("mistral"):
                    return self.fail(ln, "Mistral Large 3 spend not printed or not in its report")
                self.p(ln, m, 1, sp["mistral"][0], "Mistral Large 3 spend, run")
                self.p(ln, m, 2, sp["mistral"][1], "Mistral Large 3 spend, with the forced test")
            m = re.search(r"DeepSeek \$(\d+\.\d+) in total \(`([^`]+)`\), which already holds the \$(\d+\.\d+) spent "
                          r"before the complete-design run: the live test, the pilot \(whose replies are among the "
                          r"scored rows\) and the test and determinism calls(?:, \$(\d+\.\d+) on the run's meter)? "
                          r"\(`([^`]+)`\)", t)
            if m or sp.get("deepseek"):
                if not m or not sp.get("deepseek"):
                    return self.fail(ln, "DeepSeek spend not printed or not in its report")
                total, before = sp["deepseek"]
                self.p(ln, m, 1, total, "DeepSeek spend, total")
                self.w(ln, m, 2, "results/frontier_panel/nebius_panel.md", "DeepSeek spend, source")
                self.p(ln, m, 3, before, "DeepSeek spend, before the run")
                self.w(ln, m, 5, "results/frontier_panel/nebius_realtime_spend.md", "DeepSeek spend before the run, source")
                if m.group(4) is not None or sp.get("meter") is not None:
                    if m.group(4) is None:
                        return self.fail(ln, "the run's meter is in its report but not printed")
                    self.p(ln, m, 4, sp.get("meter"), "DeepSeek spend before the run, the real-time report's meter")
                raw = sp.get("raw")
                if raw is not None and sp.get("live") is not None:
                    # recounted from the replies: the total is every reply's cost plus the live test, and the part
                    # spent before the complete-design run (every phase but it, plus the live test) is the meter,
                    # so the total already holds it
                    self.T.printed(f"{ln.where}: DeepSeek spend, total, recounted from the replies", m.group(1),
                                   sum(raw.values()) + sp["live"])
                    before_raw = sum(v for k, v in raw.items() if k != "complete") + sp["live"]
                    self.T.printed(f"{ln.where}: DeepSeek spend before the run, recounted from the replies",
                                   m.group(3), before_raw)
                    if m.group(4) is not None:
                        self.T.printed(f"{ln.where}: DeepSeek meter, recounted from the replies", m.group(4),
                                       before_raw)
                    self.T.text(f"{ln.where}: the pilot's replies are among those the total sums", True,
                                "pilot" in raw)
        elif t.startswith("Forced-baseline test"):
            src = (F.root / "results/ladder/mistral_forced_test.md")
            a = re.search(r"Verdict: (\d+) of (\d+) forced replies pass", src.read_text(encoding="utf-8")) \
                if src.exists() else None
            src = (F.root / "results/frontier_panel/nebius_prefill_test.md")
            b = re.search(r"\| pass \((\d+)/(\d+)\) \| /v1/completions", src.read_text(encoding="utf-8")) \
                if src.exists() else None
            m = re.search(r"Mistral Large 3 passed (\d+) of (\d+) forced replies .*DeepSeek passed (\d+) of (\d+) on", t)
            if not m:
                return self.fail(ln, "forced-baseline test line not parsed")
            for gi, sm, sg in ((1, a, 1), (2, a, 2), (3, b, 1), (4, b, 2)):
                self.w(ln, m, gi, sm.group(sg) if sm else None, f"forced-baseline test, number {gi}")
        elif t.startswith("Ladders in the paper"):
            m = re.search(r"the recorded sizes (do not support that|support that|cannot settle that)", t)
            if m:
                o = F.sizes[DS]
                others = {k: v for k, v in F.sizes.items() if k in FILES and k != DS}
                below = o.get("total") and (
                    any(v.get("total") and Fraction(v["total"]) > Fraction(o["total"]) for v in others.values())
                    or any(v.get("active") and largest_active(o) is not None
                           and Fraction(v["active"]) > largest_active(o) for v in others.values()))
                self.w(ln, m, 1, "do not support that" if below else ("support that" if o.get("total") else
                                                                      "cannot settle that"), "sizes")
            for st in re.finditer(NAME_RE + r" (complete|partial|missing)\b", t):
                k = KEY_OF[st.group(1)]
                self.w(ln, st, 2, "missing" if k in F.missing else ("partial" if k in F.partial else "complete"),
                       f"{k} status")
            m = re.search(r"DeepSeek's CoT − forced \(content\), discovery draws: (.*?)\.(?: HotpotQA, pooled rows: " + V
                          + r" pp \(above (\d+) of (\d+)(?:, (\d+) tied)?\)\.)?(?= Qwen3-2507|$)", t)
            if m:
                for i in re.finditer(TASK_RE + " " + V + r" pp \(above (\d+) of (\d+)(?:, (\d+) tied)?\)", m.group(1)):
                    pl = F.place[(DS, TASK_ROW[i.group(1)], "cut", "content", "cot_minus_forced")]
                    self.p(ln, i, 2, pl["value"], f"DeepSeek {i.group(1)}", m.start(1))
                    self.p(ln, i, 3, pl["below"], f"DeepSeek {i.group(1)} below", m.start(1))
                    self.p(ln, i, 4, pl["n"], f"DeepSeek {i.group(1)} panel n", m.start(1))
                    self.tie(ln, i, 5, pl, f"DeepSeek {i.group(1)}", m.start(1))
                if m.group(2):
                    pl = F.place[(DS, "hotpot", "cut", "content", "cot_minus_forced")]
                    self.p(ln, m, 2, pl["value"], "DeepSeek HotpotQA")
                    self.p(ln, m, 3, pl["below"], "DeepSeek HotpotQA below")
                    self.p(ln, m, 4, pl["n"], "DeepSeek HotpotQA panel n")
                    self.tie(ln, m, 5, pl, "DeepSeek HotpotQA")
        elif t.startswith("Raised caps ("):
            m = re.match(r"^Raised caps \(" + VI + r"\) on the requested, CoT and five-shot requested "
                         r"arms\. Uncut outputs that ended at " + VI + r": (.*)\.$", t)
            if not m:
                return self.fail(ln, "raised-caps line not parsed")
            cap = yaml.safe_load((F.root / "config/decoding_ladder2_gsm8k.yaml").read_text(encoding="utf-8"))
            self.p(ln, m, 1, cap["max_new_tokens"]["zero_shot_cot"], "raised cap")
            self.p(ln, m, 2, cap["max_new_tokens"]["zero_shot_cot"], "raised cap")
            raw = {"requested": "direct_answer", "cot": "zero_shot_cot", "fs_requested": "fewshot_requested"}
            want = [(k, tk, a) for k in F.present for tk in TASKS if tk in F.x[k]["uncut"]
                    for a, n in F.x[k]["uncut"][tk].get("finish_length", {}).items() if a in raw and n]
            items = list(re.finditer(NAME_RE + " " + TASK_RE + r" (requested|cot|fs_requested) " + VI + " of " + VI,
                                     m.group(3)))
            self.T.text(f"{ln.where}: raised-caps arms", want, [(KEY_OF[i.group(1)], TASK_ROW[i.group(2)], i.group(3))
                                                               for i in items])
            for i in items:
                x = F.x[KEY_OF[i.group(1)]]["uncut"][TASK_ROW[i.group(2)]]
                self.p(ln, i, 4, x["finish_length"].get(i.group(3)), "ended at the raised cap", m.start(3))
                self.p(ln, i, 5, x["rows_by_arm"].get(raw[i.group(3)]), "rows", m.start(3))
        elif t.startswith("Share cut at the panel"):
            m = re.match(r"^Share cut at the panel's cap, largest " + V + r"%; arms with " + V + r"% or more "
                         r"cut: (.*?)\. Largest change of a quantity between the readings, every placed quantity on the "
                         r"discovery draws: " + self.LARGEST + r"(?:; leaving out the readable shares, " + self.LARGEST
                         + r")?; (.*?)\. On the panel's basis of each benchmark and score \(named per line in section 9\): "
                         r"(\d+) placements against the panel move(?: \((\d+) without the readable shares\))? and (\d+) "
                         r"family directions change\.$", t)
            if not m:
                return self.fail(ln, "share-cut line not parsed")
            raw = {"requested": "direct_answer", "cot": "zero_shot_cot", "fs_requested": "fewshot_requested"}
            sh = [(k, tk, a, Fraction(c, F.x[k]["cut"][tk]["rows_by_arm"][raw[a]])) for k in F.present for tk in TASKS
                  if tk in F.x[k]["cut"] for a, c in F.x[k]["cut"][tk].get("cut_at_panel_cap", {}).items() if a in raw]
            self.p(ln, m, 1, 100 * max(x[3] for x in sh), "largest share cut")
            self.p(ln, m, 2, 1, "listing threshold")
            want = [(k, tk, a) for k, tk, a, v in sh if v >= Fraction(1, 100)]
            items = list(re.finditer(NAME_RE + " " + TASK_RE + r" (requested|cot|fs_requested) " + V + "%", m.group(3)))
            self.T.text(f"{ln.where}: share-cut arms", sorted(want), sorted((KEY_OF[i.group(1)], TASK_ROW[i.group(2)],
                                                                              i.group(3)) for i in items))
            share = {(k, tk, a): v for k, tk, a, v in sh}
            for i in items:
                self.p(ln, i, 4, 100 * share[(KEY_OF[i.group(1)], TASK_ROW[i.group(2)], i.group(3))], "share cut",
                       m.start(3))
            self.largest_part(ln, m, 4, 5, 6, None, "largest change")
            self.T.text(f"{ln.where}: the largest change without the readable shares printed",
                        any(x[2] == "readable" for x in largest(F)[1]), m.group(7) is not None)
            if m.group(7) is not None:
                self.largest_part(ln, m, 7, 8, 9, False, "largest change without the readable shares")
            self.w(ln, m, 10, self.order_words(), "pairs that change order, and ties that separate")
            self.p(ln, m, 11, len(F.rank_list), "placements that move")
            wr = without_readable([x[2] for x in F.rank_list])
            self.T.text(f"{ln.where}: count without the readable shares printed", wr != "", m.group(12) is not None)
            if m.group(12) is not None:
                self.p(ln, m, 12, sum(1 for x in F.rank_list if x[2] != "readable"), "placements, no readable")
            self.p(ln, m, 13, len(F.fam_changes), "family directions that change")
        elif t.startswith("Equivalence of the token-level cut"):
            m = re.match(r"^Equivalence of the token-level cut: (.*?)\. Not run on the API models; .*"
                         r"it applies to (.*?), the ladder's default-kernel rung", t)
            if not m:
                return self.fail(ln, "equivalence line not parsed")
            self.equiv_list(ln, m, 1)
            self.dk_equiv_list(ln, m, 1)
            self.w(ln, m, 2, ", ".join(NAMES[k][1] for k in F.present if F.kernels.get(k, "").startswith("default")),
                   "default-kernel models")
        elif t.startswith("Mistral Large 3 ran at the raised caps"):
            m = re.search(r"\((complete|partial|missing); (.*)\)\.$", t)
            if not m:
                return self.fail(ln, "Mistral Large 3 line not parsed")
            self.state(ln, "mistral-large-2512", m.group(2), m.start(2))
        elif t.startswith("Families: every family"):
            m = re.search(r"runs Qwen's official FP8 on (\d+) cards", t)
            rp = F.root / "config/models_ladder.yaml"
            reg = yaml.safe_load(rp.read_text(encoding="utf-8"))["dataset2"]["models"] if rp.exists() else {}
            if m:
                self.p(ln, m, 1, (reg.get("qwen3.5-122b-a10b-fp8") or {}).get("tensor_parallel_size"), "cards")

    def model_info_table(self, where: str, head: list[str], rows: list[list[str]]) -> None:
        """Section 4: one row per ladder model: ladder, recorded size, route (cards), precision, kernels, status and
        benchmarks."""
        F = self.F
        self.T.text(f"{where}: rows", [NAMES[k][0] for k in FILES], [r[0] for r in rows])
        reg = yaml.safe_load((F.root / "config/models_ladder.yaml").read_text(encoding="utf-8"))["dataset2"]["models"] \
            if (F.root / "config/models_ladder.yaml").exists() else {}
        for r in rows:
            k = KEY_OF.get(r[0])
            if k is None:
                continue
            c = dict(zip(head, r))
            self.T.text(f"{where}: {k} ladder", FAMILY_OF.get(k, OUTSIDE), c["Ladder"])
            ln = Line(c["Size (recorded)"], f"{where}: {k} size")
            ln.mask_structural()
            m = re.match(r"^(.*)$", ln.text)
            self.size_cell(ln, m, 1, k)
            self.finish(ln)
            ln = Line(c["Route"], f"{where}: {k} route")
            ln.mask_structural()
            for m in re.finditer(r"(\d+) (?:reserved )?RTX PRO 6000|tensor parallel (\d+)", ln.text):
                gi = 1 if m.group(1) else 2
                self.p(ln, m, gi, (reg.get(k) or {}).get("tensor_parallel_size"), f"{k} cards")
            self.finish(ln)
            ln = Line(c["Precision"], f"{where}: {k} precision")
            ln.mask_structural()
            self.finish(ln)
            self.T.text(f"{where}: {k} kernels", kernels(F.root, k), c["Kernels"])
            self.T.text(f"{where}: {k} status", "missing" if k in F.missing else ("partial" if k in F.partial else
                                                                                  "complete"), c["Status"].split(":")[0])
            self.T.text(f"{where}: {k} benchmarks", ", ".join(TASK_NAME[tk] for tk in TASKS if tk in F.x[k]["cut"]
                                                              or tk in F.x[k]["uncut"]) or "none", c["Benchmarks in"])

    def completeness_table(self, where: str, rows: list[list[str]]) -> None:
        """Section 4: per model and benchmark, the design rows and arms, the scored rows, the id problems, unscored rows,
        matching (arm, configuration) cells and whether the request files equal the API runs'."""
        F = self.F
        self.T.text(f"{where}: rows", [(NAMES[k][0], TASK_NAME[tk]) for k in FILES for tk in TASKS],
                    [(r[0], r[1]) for r in rows])
        for r in rows:
            k, tk = KEY_OF.get(r[0]), TASK_ROW.get(r[1])
            if k is None or tk is None:
                continue
            x = F.comp[k]["tasks"].get(tk)
            w = f"{where}: {k} {tk}"
            if x is None:
                self.T.text(f"{w} design absent", True, r[2].startswith("design file absent"))
                continue
            m = re.match(r"^" + VI + r" \((\d+)\)$", r[2])
            if not m:
                self.T.n += 1
                self.T.bad.append(f"{w}: design cell not parsed: {r[2]!r}")
                continue
            self.T.printed(f"{w} design rows", m.group(1), x["design_rows"])
            self.T.printed(f"{w} arms", m.group(2), x["arms"])
            cu, un = x.get("cut") or {}, x.get("uncut") or {}
            self.T.printed(f"{w} scored cut", r[3], cu.get("rows", 0))
            self.T.printed(f"{w} scored uncut", r[4], un.get("rows", 0))
            for col, f in ((5, "missing"), (6, "duplicate"), (7, "extra"), (8, "differ"), (9, "unscored")):
                a, b = cu.get(f, 0), un.get(f, 0)
                parts = r[col].split(" / ")
                if a == b:
                    self.T.printed(f"{w} {f}", r[col], a)
                elif len(parts) == 2:
                    self.T.printed(f"{w} {f} cut", parts[0], a)
                    self.T.printed(f"{w} {f} uncut", parts[1], b)
                else:
                    self.T.text(f"{w} {f}", f"{a:,} / {b:,}", r[col])
            m = re.match(r"^(\d+) of (\d+)$", r[10])
            if not m:
                self.T.n += 1
                self.T.bad.append(f"{w}: cells not parsed: {r[10]!r}")
            else:
                self.T.printed(f"{w} cells matching", m.group(1), cu.get("cells_matching", 0))
                self.T.printed(f"{w} cells", m.group(2), x["cells"])
            self.T.text(f"{w} design equals the API runs'", "it is that file" if (k in API or k == API_DESIGN_KEY) else
                        ("yes" if x.get("equals_api") else "no"), r[11])

    def sec4_line(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.match(r"^- " + NAME_RE + r": (.*)\.$", t)
        if m:
            return self.state(ln, KEY_OF[m.group(1)], m.group(2), m.start(2))
        m = re.match(r"^Panel rungs of the family ladders, sizes as recorded: (.*)\. Rungs are ordered by total "
                     r"parameters \(section 2\)\.$", t)
        if m:
            keys = [k for f in FAMILIES.values() for k, src in f["rungs"] if src == "panel"]
            ks = "|".join(re.escape(k) for k in keys)
            items = list(re.finditer(r"(" + ks + r") (.*?)(?=; (?:" + ks + r") |$)", m.group(1)))
            self.T.text(f"{ln.where}: panel rungs listed", keys, [i.group(1) for i in items])
            for i in items:
                self.size_cell(ln, i, 2, i.group(1), off=m.start(1))
            self.T.text(f"{ln.where}: order of the rungs", True, self.ordered())
            return
        m = re.match(r"^Shortfalls among the models with scored rows \(arm × configuration cells, missing arms\): "
                     r"(none|\d+)\.$", t)
        if m:
            n = sum((x.get(r) or {}).get("cells_short", 0) + bool((x.get(r) or {}).get("missing_arms"))
                    for k in F.present for x in F.comp[k]["tasks"].values() if x for r in READINGS)
            if m.group(1) == "none":
                self.T.text(f"{ln.where}: shortfalls", 0, n)
            else:
                self.p(ln, m, 1, n, "shortfalls")

    def marks_line(self, ln: Line, tk: str) -> None:
        """Section 6: the panel's own family members on one benchmark, CoT − forced (content), with their placement
        among the panel (themselves left out of the tie count)."""
        F = self.F
        m = re.match(r"^Panel family members, CoT − forced \(content\), (discovery draws|pooled rows): (.*)\.$", ln.text)
        if not m:
            return self.fail(ln, "panel family members line not parsed")
        self.w(ln, m, 1, BASIS_WORDS[basis_of(tk, "content")], "basis")
        col = F.panel[tk]["cot_minus_forced_content"]
        vals = {k: v for k, v in col.items() if not (isinstance(v, float) and math.isnan(v))}
        keys = [k for f in FAMILIES.values() for k in [r for r, s_ in f["rungs"] if s_ == "panel"] + f["context"]
                if k in vals]
        items = list(re.finditer(alt(PANEL_RUNGS) + " " + V + r" \(above (\d+)/(\d+)(?:, (\d+) tied)?\)", m.group(2)))
        self.T.text(f"{ln.where}: panel members listed", keys, [i.group(1) for i in items])
        for i in items:
            k, v = i.group(1), vals.get(i.group(1))
            if v is None:
                continue
            o = m.start(2)
            self.p(ln, i, 2, v, f"{k} value", o)
            self.p(ln, i, 3, sum(float(x) < float(v) - TOL for x in vals.values()), f"{k} below", o)
            self.p(ln, i, 4, len(vals), f"{k} panel n", o)
            eq = sum(abs(float(x) - float(v)) <= TOL for j, x in vals.items() if j != k)
            if i.group(5) is None:
                self.T.printed(f"{ln.where}: {k} tied", "0", eq)
            else:
                self.p(ln, i, 5, eq, f"{k} tied", o)

    def template_table(self, where: str, rows: list[list[str]]) -> None:
        """Section 7, template SD: C, and per quantity the SD over configurations (rounded half-even on the exact root of
        the exact variance, checked here by bounds) and, in brackets, the variance minus the item sampling."""
        F = self.F
        want = [(k, tk, s_) for k in F.present for tk in TASKS if tk in F.x[k]["cut"] for s_ in SCORE_COL]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1]), r[2]) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        qs = ("requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced")
        for r, (k, tk, s_) in zip(rows, got):
            if k is None or tk is None or s_ not in SCORE_COL:
                continue
            x = F.x[k]["cut"][tk]
            self.T.printed(f"{where}: {k} {tk} {s_} C", r[3], x["configs_answered"])
            for q, c in zip(qs, r[4:9]):
                m = re.match(r"^(\S+) \[(\S+)\]$", c)
                tm = x["template"][s_][q]
                if not m:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: {k} {tk} {s_} {q}: not of the form SD [excess]: {c!r}")
                    continue
                self.T.root(f"{where}: {k} {tk} {s_} {q} SD", m.group(1), tm["V"])
                self.T.printed(f"{where}: {k} {tk} {s_} {q} variance minus item sampling", m.group(2), tm["excess"])

    def edge(self, ln: Line) -> None:
        """Section 12, edge cases."""
        t, F = ln.text, self.F
        m = re.match(r"^" + NAME_RE + r": template variance minus the item sampling \(interaction mean square over the "
                     r"items\) is negative in (\d+) of (\d+) cells \(benchmark × score × quantity, cut\)", t)
        if m:
            k = KEY_OF[m.group(1)]
            cells = [c for tk in TASKS if tk in F.x[k]["cut"] for s_ in SCORE_COL
                     for c in F.x[k]["cut"][tk]["template"][s_].values()]
            self.p(ln, m, 2, sum(c["excess"] is not None and c["excess"] < 0 for c in cells), f"{k} negative cells")
            self.p(ln, m, 3, len(cells), f"{k} cells")
            return
        m = re.match(r"^" + NAME_RE + ": " + TASK_RE + r" \((cut|uncut)\): (\d+) rows end with a finish reason other "
                     r"than stop or length \((.*)\); scored as they came$", t)
        if m:
            k, tk, r = KEY_OF[m.group(1)], TASK_ROW[m.group(2)], m.group(3)
            odd = F.x[k][r][tk].get("finish_other", {})
            self.p(ln, m, 4, sum(odd.values()), f"{k} {tk} {r} other finish reasons")
            items = list(re.finditer(r"([\w-]+) in (\d+) ([\w-]+)", m.group(5)))
            self.T.text(f"{ln.where}: finish reasons", sorted((a, fr) for a, fr in odd),
                        sorted((i.group(3), i.group(1)) for i in items))
            for i in items:
                self.p(ln, i, 2, odd.get((i.group(3), i.group(1))), f"{k} {tk} {r} {i.group(1)} {i.group(3)}",
                       m.start(5))
            return
        m = re.match(r"^" + NAME_RE + ": " + TASK_RE + r" \((cut|uncut)\): rows per arm differ from the design (.*)$", t)
        if m:
            k, tk, r = KEY_OF[m.group(1)], TASK_ROW[m.group(2)], m.group(3)
            rows = F.x[k][r][tk]["rows_by_arm"]
            want = [a for a, n in DESIGN[tk].items() if rows.get(a, 0) != n]
            items = list(re.finditer(r"([a-z_]+) " + VI + " of " + VI, m.group(4)))
            self.T.text(f"{ln.where}: arms off the design", want, [i.group(1) for i in items])
            for i in items:
                self.p(ln, i, 2, rows.get(i.group(1), 0), f"{k} {tk} {r} {i.group(1)} rows", m.start(4))
                self.p(ln, i, 3, DESIGN[tk].get(i.group(1)), f"{k} {tk} {r} {i.group(1)} design rows", m.start(4))
            return
        m = re.match(r"^" + NAME_RE + ": " + TASK_RE + r" \((cut|uncut)\): (\d+) zero-shot rows left out because their "
                     r"unit lacks an arm$", t)
        if m:
            x = F.x[KEY_OF[m.group(1)]][m.group(3)][TASK_ROW[m.group(2)]]
            self.p(ln, m, 4, x["zero_left_out"], "zero-shot rows left out")
            return
        m = re.match(r"^" + NAME_RE + ": " + TASK_RE + r" \((cut|uncut)\): (\d+) of (\d+) configurations; draw means over "
                     r"a missing configuration are absent$", t)
        if m:
            x = F.x[KEY_OF[m.group(1)]][m.group(3)][TASK_ROW[m.group(2)]]
            self.p(ln, m, 4, x["configs_answered"], "configurations answered")
            self.p(ln, m, 5, x["configs_design"], "configurations of the design")
            return
        m = re.match(r"^panel (\S+): (\S+) = (\S+) pp between the panel's files$", t)
        if m:
            v = (F.panel_checks["named"].get(m.group(1)) or {}).get(m.group(2))
            self.w(ln, m, 3, None if v is None else f"{v:.3g}", f"panel files {m.group(1)} {m.group(2)}")
            return
        m = re.match(r"^Ties with panel models: (\d+) placement cells? in the cut reading and (\d+) in the uncut reading "
                     r"equal a panel value", t)
        if m:
            cut = sum(1 for key, pl in F.place.items() if key[2] == "cut" and pl.get("equal"))
            cut += sum(1 for d in (F.bbh_gap, F.fs_place) for pl in d.values() if pl.get("equal"))
            unc = sum(1 for key, pl in F.place.items() if key[2] == "uncut" and pl.get("equal"))
            self.p(ln, m, 1, cut, "ties with panel models, cut")
            self.p(ln, m, 2, unc, "ties with panel models, uncut")
            return
        m = re.match(r"^Ties between ladder models: (\d+) pairs? tied exactly in one reading", t)
        if m:
            self.p(ln, m, 1, len(F.tie_list), "ties between ladder models")
            return

    def sources(self, ln: Line) -> None:
        t, F = ln.text, self.F
        m = re.search(r"\(largest \|difference\| (\S+) pp over (\d+) checks;", t)
        if m:
            self.w(ln, m, 1, f"{F.panel_checks['max']:.1g}", "panel files: largest difference")
            self.p(ln, m, 2, F.panel_checks["n"], "panel files: checks")
            return
        m = re.match(r"^Runtime (\d+) s\.$", t)
        if m:
            self.w(ln, m, 1, f"{F.J.get('runtime_s', float('nan')):.0f}", "runtime")

    def sec11(self, ln: Line) -> None:
        m = re.match(r"^(\d+) values \(.*\) compared with `[^`]+`: (\d+) differ by more than (1e-0?9)\.$", ln.text)
        if m:
            rc = reproduction_count(self.F)
            if rc is None:
                return self.fail(ln, "no reproduction here")
            self.p(ln, m, 1, rc[0], "reproduction values compared")
            self.p(ln, m, 2, rc[1], "reproduction values differing")
            self.T.num(f"{ln.where}: reproduction tolerance", float(m.group(3)), TOL)
            ln.use(*m.span(3))

    def wn_family_para(self, ln: Line) -> None:
        """The note's rung list: each rung's source and precision, and the panel's precision."""
        t, F = ln.text, self.F
        m = re.search(r"The panel ran in (.*) \(footnote F3\)\.$", t)
        if m:
            self.w(ln, m, 1, self.precision_phrase(), "the panel's precision")
        for name, f in FAMILIES.items():
            for k, src in f["rungs"]:
                if src != "panel":
                    continue
                mm = re.search(re.escape(SIZE[k]) + r" \(panel, ([^)]*)\)", t[t.index(name + ":"):] if name + ":" in t
                               else "")
                if mm:
                    self.T.text(f"{ln.where}: {k} precision", F.prec.get(k), mm.group(1))

    def family_heading(self, ln: Line) -> None:
        m = re.match(r"^### " + FAM_RE + r": (.*) \((\d+) of (\d+) rungs in\)$", ln.text)
        if not m:
            return
        f = self.F.fam[m.group(1)]
        self.w(ln, m, 2, " → ".join(SIZE[r["key"]] for r in f["rungs"]), "rung labels")
        self.p(ln, m, 3, len(f["present"]), "rungs in")
        self.p(ln, m, 4, len(f["rungs"]), "rungs")

    def table(self, sec: str, where: str, h3: str, label: str, head: list[str], rows: list[list[str]]) -> bool:
        """Every table of ladder.md to its checker; False for a table no checker reads."""
        if sec == "1":
            self.glance(where, head, rows)
            return True
        if sec == "4":
            if head == ["Model", "Ladder", "Size (recorded)", "Route", "Precision", "Kernels", "Status", "Benchmarks in"]:
                self.model_info_table(where, head, rows)
                return True
            if head[:2] == ["Model", "Benchmark"] and head[2].startswith("Design rows"):
                self.completeness_table(where, rows)
                return True
        if sec == "5":
            fm = re.match(FAM_RE + ":", h3)
            if fm and head[0] == "Quantity":
                self.family_table(where, fm.group(1), label, head, rows)
                return True
            if fm and head[0] == "Panel model":
                self.context_table(where, fm.group(1), rows)
                return True
        if sec == "6":
            self.placement_table(where, h3, head, rows)
            return True
        if sec in ("7", "8"):
            reading = "cut" if sec == "7" else "uncut"
            if head[3:4] == ["Requested (48 draws)"]:
                self.model_table(where, reading, head, rows)
                return True
            if sec == "7" and head[3:4] == ["C"]:
                self.template_table(where, rows)
                return True
            if head[2:4] == ["Arm", "Reading"]:
                self.fewshot_table(where, rows)
                return True
            if head[2:3] and head[2].startswith("Readable (%)"):
                self.readable_table(where, rows)
                return True
        if sec == "9":
            if head[2:4] == ["Arm", "Rows"]:
                self.cut_table(where, rows)
                return True
            if head[2:4] == ["Score", "Quantity"]:
                self.change_table(where, rows, label)
                return True
        return False

    # -- note, section "Orderings among the ladder models" (scripts/ladder_orderings.py) -----------------------------
    ORD_COUNT_RE = re.compile(r"(?:" + TASK_RE + r"|(?P<pool>pooled over the first three,)) (?P<k>\d{1,3}(?:,\d{3})*) of "
                              r"(?P<n>\d{1,3}(?:,\d{3})*)(?: \((?P<pct>\d+)%(?:, (?P<lo>-?\d\.\d\d) to (?P<hi>-?\d\.\d\d))?\))?")

    def ord_counts(self, ln: Line, a: int, b: int, blk: dict, what: str, ci=None) -> None:
        """'<benchmark> k of n (p%, lo to hi)' and 'pooled over the first three, k of n ...' in ln.text[a:b]: every
        benchmark and the pooled value once, each against this checker's counts (the interval: the file's, which
        check_orderings holds within ORD_BOOT_TOL of this checker's own)."""
        seen = []
        for m in self.ORD_COUNT_RE.finditer(ln.text[a:b]):
            tk = "pooled" if m.group("pool") else TASK_ROW[m.group(1)]
            seen.append(tk)
            x = blk[tk]
            self.p(ln, m, "k", x["reversed"], f"{what} {tk} reversed", a)
            self.p(ln, m, "n", x["pairs"], f"{what} {tk} pairs", a)
            if m.group("pct"):
                self.p(ln, m, "pct", round(Fraction(100 * x["reversed"], x["pairs"])), f"{what} {tk} percent", a)
            if m.group("lo"):
                c = ci(tk) if ci else None
                if not c:
                    self.fail(ln, f"{what} {tk}: an interval is printed, none is known here")
                    ln.use(a + m.start("lo"), a + m.end("hi"))
                else:
                    self.p(ln, m, "lo", Fraction(c[0]), f"{what} {tk} bootstrap low", a)
                    self.p(ln, m, "hi", Fraction(c[1]), f"{what} {tk} bootstrap high", a)
        self.T.text(f"{ln.where}: {what} benchmarks printed", TASKS + ["pooled"], seen)

    def orderings(self, ln: Line, is_b: bool) -> None:
        F = self.F
        O, Jo = getattr(F, "orderings", None), getattr(F, "ord_json", None)
        t = ln.text
        if not is_b:
            return                       # the opening paragraph; finish() catches any number in it
        if O is None or Jo is None:
            self.fail(ln, "a line of the orderings section, but no ladder_orderings.json to check it against")
            return
        RV = Jo.get("reversals") or {}

        def ci_of(*path):
            return lambda tk: g(RV, *path, tk, "model_bootstrap", "ci95")

        def split(sep):
            i = t.find(sep)
            return (i, i + len(sep)) if i >= 0 else (len(t), len(t))

        GAP_GROUPS = {"Six ladder models": O["six"]["cut"]["content"]["discovery"],
                      "Models of 70B and above": O["big"]["cut"]["content"], "Combined panel": O["comb"]["cut"]["content"]}
        mg = re.match(r"(Six ladder models|Models of 70B and above|Combined panel), content, gaps above (\d+) pp: ", t)
        if t.startswith("Six ladder models, content: "):
            i, j = split(". Strict: ")
            self.ord_counts(ln, 0, i, O["six"]["cut"]["content"]["discovery"], "six, content")
            self.ord_counts(ln, j, len(t), O["six"]["cut"]["strict"]["discovery"], "six, strict")
        elif mg:
            # reversed of the pairs whose two gaps both exceed 1, 2 and 5 pp
            blk = GAP_GROUPS[mg.group(1)]
            heads = [(mg, 2)] + [(x, 1) for x in re.finditer(r"\. Above (\d+) pp: ", t)]
            ths = []
            for n_, (hm, gi) in enumerate(heads):
                th = int(hm.group(gi))
                ths.append(th)
                self.p(ln, hm, gi, th, f"{mg.group(1)} gap threshold {n_}")
                a, b = hm.end(), (heads[n_ + 1][0].start() if n_ + 1 < len(heads) else len(t))
                if th not in ORD_GAPS:
                    self.fail(ln, f"gap threshold {th} pp is not one this checker counts")
                    continue
                self.ord_counts(ln, a, b, {tk: blk[tk]["above"][th] for tk in TASKS + ["pooled"]},
                                f"{mg.group(1)}, gaps above {th} pp")
            self.T.text(f"{ln.where}: {mg.group(1)} gap thresholds printed", [1, 2, 5], ths)
        elif t.startswith("Reversed pairs of the six whose gap under either gain is "):
            m = re.match(r"Reversed pairs of the six whose gap under either gain is (\d+) pp or less ", t)
            if not m:
                self.fail(ln, "sub-1 pp line not in its form")
                return
            self.p(ln, m, 1, ORD_SUB_PP, "the named pairs' gap")
            six = O["six"]["cut"]["content"]["discovery"]
            want = sorted((tk, p_) for tk in TASKS for p_, (d1, d2) in six[tk]["gaps"].items()
                          if min(d1, d2) <= ORD_SUB_PP)
            got = []
            for mm in re.finditer(TASK_RE + ", " + NAME_RE + " and " + NAME_RE + r" \((\d+\.\d\d), (\d+\.\d\d)\)", t):
                tk, ka, kb = TASK_ROW[mm.group(1)], KEY_OF[mm.group(2)], KEY_OF[mm.group(3)]
                d = six[tk]["gaps"].get((ka, kb)) or six[tk]["gaps"].get((kb, ka))
                got.append((tk, (ka, kb) if (ka, kb) in six[tk]["gaps"] else (kb, ka)))
                if d is None:
                    self.fail(ln, f"{tk}: {ka} and {kb} named, but they are not reversed here")
                    ln.use(mm.start(4), mm.end(5))
                    continue
                self.p(ln, mm, 4, d[0], f"{tk} {ka} {kb} gap, CoT - requested")
                self.p(ln, mm, 5, d[1], f"{tk} {ka} {kb} gap, CoT - forced")
            self.T.text(f"{ln.where}: the reversed pairs with a gap of {ORD_SUB_PP} pp or less", want, sorted(got))
        elif t.startswith("The hand counts"):
            seen = []
            for m in re.finditer(TASK_RE + r" (?P<w>\d+) of (?P<n>\d+) \((?P<h>\d+) here\)", t):
                tk = TASK_ROW[m.group(1)]
                seen.append(tk)
                w = O["hand"]["file"].get(tk)
                self.p(ln, m, "w", w[0] if w else None, f"hand count {tk}")
                self.p(ln, m, "n", O["six"]["cut"]["content"]["all_draws"][tk]["pairs"], f"pairs {tk}")
                self.p(ln, m, "h", O["six"]["cut"]["content"]["all_draws"][tk]["reversed"], f"48-draw count {tk}")
            self.T.text(f"{ln.where}: hand counts printed", TASKS, seen)
            i, j = split("On the discovery draws: ")
            e = t.find(". ", j)
            e = len(t) if e < 0 else e
            seen = []
            for m in re.finditer(TASK_RE + r" (?P<d>\d+)", t[j:e]):
                tk = TASK_ROW[m.group(1)]
                seen.append(tk)
                self.p(ln, m, "d", O["six"]["cut"]["content"]["discovery"][tk]["reversed"], f"discovery count {tk}", j)
            self.T.text(f"{ln.where}: discovery counts printed", TASKS, seen)
            V2 = r"(-?\d+\.\d\d)"
            rx = (TASK_RE + ", " + NAME_RE + " and " + NAME_RE + r", (reversed|not reversed) on the discovery draws only "
                  r"\(CoT − requested " + V2 + " and " + V2 + " on the 48 draws, " + V2 + " and " + V2
                  + r" on the discovery draws; CoT − forced " + V2 + " and " + V2 + r"\)")
            got = {}
            for m in re.finditer(rx, t):
                tk = TASK_ROW[m.group(1)]
                ka, kb = KEY_OF[m.group(2)], KEY_OF[m.group(3)]
                got.setdefault(tk, []).append(tuple(sorted((ka, kb))))
                on_disc = (ka, kb) in O["six"]["cut"]["content"]["discovery"][tk]["rev_set"] or \
                    (kb, ka) in O["six"]["cut"]["content"]["discovery"][tk]["rev_set"]
                self.w(ln, m, 4, "reversed" if on_disc else "not reversed", f"status {tk} {ka} {kb}")
                x = F.x
                for grp, k, basis, q in ((5, ka, "all_draws", "cot_minus_requested"),
                                         (6, kb, "all_draws", "cot_minus_requested"),
                                         (7, ka, "discovery", "cot_minus_requested"),
                                         (8, kb, "discovery", "cot_minus_requested"),
                                         (9, ka, "discovery", "cot_minus_forced"), (10, kb, "discovery", "cot_minus_forced")):
                    self.p(ln, m, grp, x[k]["cut"][tk]["content"][basis][q], f"{tk} {k} {basis} {q}")
            for tk in TASKS:
                self.T.text(f"{ln.where}: pairs differing between the bases, {tk}",
                            sorted(tuple(sorted(p)) for p in O["hand"]["diff"][tk]), sorted(got.get(tk, [])))
            # where the hand count and the 48-draw count differ, the 48-draw pairs, named with gaps
            k48 = t.find("the pairs reversed on the 48 draws")
            seen48 = []
            if k48 >= 0:
                for mt in re.finditer(TASK_RE + r" reverses (\d+): ", t[k48:]):
                    tk = TASK_ROW[mt.group(1)]
                    seen48.append(tk)
                    a48 = O["six"]["cut"]["content"]["all_draws"][tk]
                    self.p(ln, mt, 2, a48["reversed"], f"48-draw reversed {tk}", k48)
                    a0 = k48 + mt.end()
                    e0 = t.find("; ", a0)
                    e0 = len(t) if e0 < 0 else e0
                    named = []
                    for mm in re.finditer(NAME_RE + " and " + NAME_RE + r" \((\d+\.\d\d), (\d+\.\d\d)\)", t[a0:e0]):
                        ka, kb = KEY_OF[mm.group(1)], KEY_OF[mm.group(2)]
                        p_ = (ka, kb) if (ka, kb) in a48["gaps"] else (kb, ka)
                        named.append(p_)
                        d = a48["gaps"].get(p_)
                        if d is None:
                            self.fail(ln, f"48 draws, {tk}: {ka} and {kb} named, but they are not reversed here")
                            ln.use(a0 + mm.start(3), a0 + mm.end(4))
                            continue
                        self.p(ln, mm, 3, d[0], f"48 draws {tk} {ka} {kb} gap, CoT - requested", a0)
                        self.p(ln, mm, 4, d[1], f"48 draws {tk} {ka} {kb} gap, CoT - forced", a0)
                    self.T.text(f"{ln.where}: 48-draw reversed pairs named, {tk}", sorted(a48["gaps"]), sorted(named))
            want48 = [tk for tk in TASKS if O["hand"]["file"].get(tk) and
                      O["hand"]["file"][tk][0] != O["six"]["cut"]["content"]["all_draws"][tk]["reversed"]]
            self.T.text(f"{ln.where}: benchmarks whose 48-draw pairs are named", want48, seen48)
        elif t.startswith("Combined panel of the "):
            m = re.match(r"Combined panel of the (\d+) panel models and the six \((\d+) models; (\d+) on BIG-Bench Hard\), "
                         r"(content|strict), reversed pairs \(share, model bootstrap (\d+)% interval\): ", t)
            if not m:
                self.fail(ln, "combined-panel line not in its form")
                return
            s = m.group(4)
            self.p(ln, m, 1, O["panel"][s]["gsm8k"]["models"], "panel models")
            self.p(ln, m, 2, O["comb"]["cut"][s]["gsm8k"]["models"], "combined models")
            self.p(ln, m, 3, O["comb"]["cut"][s]["bbh"]["models"], "combined models, BIG-Bench Hard")
            self.p(ln, m, 5, 95, "interval level")
            self.ord_counts(ln, m.end(), len(t), O["comb"]["cut"][s], f"combined, {s}", ci_of("combined", "cut", s))
        elif t.startswith("The ") and " panel models (" in t[:40]:
            m = re.match(r"The (\d+) panel models \((\d+) on BIG-Bench Hard\), (content|strict), the same: ", t)
            if not m:
                self.fail(ln, "panel line not in its form")
                return
            s = m.group(3)
            self.p(ln, m, 1, O["panel"][s]["gsm8k"]["models"], "panel models")
            self.p(ln, m, 2, O["panel"][s]["bbh"]["models"], "panel models, BIG-Bench Hard")
            self.ord_counts(ln, m.end(), len(t), O["panel"][s], f"panel, {s}", ci_of("panel", s))
        elif t.startswith("Models of 70B and above ("):
            names = [KEY_OF[x.group(0)] for x in re.finditer(NAME_RE, t[:t.find("), content: ")])]
            self.T.text(f"{ln.where}: models of 70B and above", sorted(BIG), sorted(names))
            i0 = t.find("), content: ") + len("), content: ")
            i, j = split(". Strict: ")
            # every member by its own recorded total (DeepSeek: V4.1-Flash's published size,
            # taken for the 1017 checkpoint), each of 70B or more
            cav = re.search(r"\. Each is (\d+)B or more in total parameters by its recorded size \(`ladder\.md` section "
                            r"4\): (.*)\.$", t)
            self.ord_counts(ln, i0, i, O["big"]["cut"]["content"], "70B and above, content")
            self.ord_counts(ln, j, cav.start() + 1 if cav else len(t), O["big"]["cut"]["strict"], "70B and above, strict")
            if not cav:
                self.fail(ln, "the 70B-and-above line does not give its members' recorded sizes")
            else:
                self.w(ln, cav, 1, "70", "70B group: its threshold")
                want = ", ".join(f"{NAMES[k][1]} {F.sizes[k].get('total')}B"
                                 + (f" ({F.sizes[k]['caveat']})" if F.sizes[k].get("caveat") else "") for k in BIG)
                self.w(ln, cav, 2, want, "70B group: each member's recorded total")
                for k in FILES:
                    tk = F.sizes[k].get("total")
                    self.T.text(f"{ln.where}: {k} in the group of 70B and above", k in BIG,
                                tk is not None and Fraction(tk) >= 70)
        elif t.startswith("w by family, discovery draws: "):
            pos = sorted((t.find(f"{fam} ("), fam) for fam in FAMILIES)
            ds = t.find(NAMES[DS][1] + ": ")
            rg = t.find("The panel's range: ")
            if min(p for p, _ in pos) < 0 or ds < 0 or rg < 0:
                self.fail(ln, "w line not in its form")
                return
            for n_, (a, fam) in enumerate(pos):
                b = pos[n_ + 1][0] if n_ + 1 < len(pos) else ds
                seen = []
                for m in re.finditer(TASK_RE + r" (?P<v>\d\.\d{3}(?: → \d\.\d{3})+) \(" + DIRW + r"\)", t[a:b]):
                    tk = TASK_ROW[m.group(1)]
                    seen.append(tk)
                    f = O["w"]["fam"][fam][tk]
                    a0 = a + m.start("v")
                    parts = m.group("v").split(" → ")
                    if len(parts) != len(f["values"]):
                        self.fail(ln, f"w {fam} {tk}: {len(parts)} values printed, {len(f['values'])} rungs")
                    for i_, (s_, v) in enumerate(zip(parts, f["values"])):
                        self.T.printed(f"{ln.where}: w {fam} {tk} rung {i_}", s_, Fraction(v) if isinstance(v, float)
                                       else v)
                    ln.use(a0, a0 + len(m.group("v")))
                    self.w(ln, m, 3, f["direction"], f"w direction {fam} {tk}", a)
                self.T.text(f"{ln.where}: w {fam} benchmarks printed", TASKS, seen)
            seen = []
            for m in re.finditer(TASK_RE + r" (?P<v>\d\.\d{3})", t[ds:rg]):
                tk = TASK_ROW[m.group(1)]
                seen.append(tk)
                self.p(ln, m, "v", O["w"]["ladder"][DS][tk], f"w DeepSeek {tk}", ds)
            self.T.text(f"{ln.where}: w DeepSeek benchmarks printed", TASKS, seen)
            seen = []
            for m in re.finditer(TASK_RE + r" (?P<lo>\d\.\d{3}) to (?P<hi>\d\.\d{3})", t[rg:]):
                tk = TASK_ROW[m.group(1)]
                seen.append(tk)
                pw = O["w"]["panel"][tk]
                self.p(ln, m, "lo", pw["min"], f"w panel min {tk}", rg)
                self.p(ln, m, "hi", pw["max"], f"w panel max {tk}", rg)
            self.T.text(f"{ln.where}: w panel benchmarks printed", TASKS, seen)
        elif t.startswith("Baseline shift (requested − forced), "):
            m = re.match(r"Baseline shift \(requested − forced\), (content|strict), discovery draws, with its part from "
                         r"rows whose requested output is longer than (\d+) tokens and its part from the other rows: ", t)
            if not m:
                self.fail(ln, "shift line not in its form")
                return
            s = m.group(1)
            self.p(ln, m, 2, SPLIT_TOKENS, "split tokens")
            starts = [(x.start(), KEY_OF[x.group(0)]) for x in re.finditer(NAME_RE, t) if x.start() >= m.end()]
            self.T.text(f"{ln.where}: shift models printed", list(FILES), [k for _, k in starts])
            for n_, (a, k) in enumerate(starts):
                b = starts[n_ + 1][0] if n_ + 1 < len(starts) else len(t)
                seen = []
                for mm in re.finditer(TASK_RE + r" (-?\d+\.\d) \(long (-?\d+\.\d), short (-?\d+\.\d)\)", t[a:b]):
                    tk = TASK_ROW[mm.group(1)]
                    seen.append(tk)
                    x = O["shift"]["cut"][s][k][tk]
                    self.p(ln, mm, 2, x["shift"], f"shift {s} {k} {tk}", a)
                    self.p(ln, mm, 3, x["long"], f"shift long {s} {k} {tk}", a)
                    self.p(ln, mm, 4, x["short"], f"shift short {s} {k} {tk}", a)
                self.T.text(f"{ln.where}: shift {k} benchmarks printed", TASKS, seen)
        elif t.startswith(NAMES[QWEN][1] + ", zero-shot requested arm"):
            m = re.match(re.escape(NAMES[QWEN][1]) + r", zero-shot requested arm, all rows: replies the strict rule cannot "
                         r"read (.*?); of these, the bare answer with no box (.*?) \(median length (.*?) tokens\)\.", t)
            Q = O.get("qwen")
            if not m or not Q:
                self.fail(ln, "Qwen line not in its form, or no raw replies to read it against")
                return
            for grp, rx, chk in ((1, r" (\d{1,3}(?:,\d{3})*) of (\d{1,3}(?:,\d{3})*)", (("unreadable", 2), ("rows", 3))),
                                 (2, r" (\d{1,3}(?:,\d{3})*)", (("bare", 2),)), (3, r" (\d+)", (("median", 2),))):
                a = m.start(grp)
                seen = []
                for mm in re.finditer(TASK_RE + rx, m.group(grp)):
                    tk = TASK_ROW[mm.group(1)]
                    seen.append(tk)
                    for f_, gi in chk:
                        v = Q[tk][f_]
                        self.p(ln, mm, gi, round(v) if f_ == "median" else v, f"Qwen {f_} {tk}", a)
                self.T.text(f"{ln.where}: Qwen benchmarks printed ({grp})", TASKS, seen)
        elif t.startswith("Uncut reading (Supplementary Information), content: six ladder models "):
            a = len("Uncut reading (Supplementary Information), content: six ladder models ")
            i, j = split("; combined panel ")
            self.ord_counts(ln, a, i, O["six"]["uncut"]["content"]["discovery"], "six, uncut, content")
            self.ord_counts(ln, j, len(t), O["comb"]["uncut"]["content"], "combined, uncut, content",
                            ci_of("combined", "uncut", "content"))
        else:
            self.fail(ln, "a line of the orderings section that no handler reads")

    def headline(self, where: str, head: list[str], rows: list[list[str]]) -> None:
        F = self.F
        want = [(k, tk) for k in F.present for tk in TASKS if tk in F.x[k]["cut"]]
        got = [(KEY_OF.get(r[0]), TASK_ROW.get(r[1])) for r in rows]
        self.T.text(f"{where}: rows", want, got)
        for r, (k, tk) in zip(rows, got):
            if k is None or tk is None:
                continue
            x = F.x[k]["cut"][tk]
            for h, c in zip(head[2:], r[2:]):
                hm = re.match(r"^(.*) \((48 draws|discovery draws)\)$", h)
                if not hm or hm.group(1) not in Q_OF:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: column not parsed: {h!r}")
                    continue
                q, basis = Q_OF[hm.group(1)], ("all_draws" if hm.group(2) == "48 draws" else "discovery")
                if q == "w":
                    self.T.printed(f"{where}: {k} {tk} w", c, x["w"][basis])
                    continue
                parts = c.split(" / ")
                if len(parts) != 2:
                    self.T.n += 1
                    self.T.bad.append(f"{where}: {k} {tk} {h}: not strict / content: {c!r}")
                    continue
                for s, part in zip(SCORE_COL, parts):
                    self.T.printed(f"{where}: {k} {tk} {h} {s}", part, value(x, s, basis, q))


def dk_caveat(dkf: dict) -> str:
    """F8's caveat for a default-kernel check on file, written here from its statement: the rows were
    chosen because the run cut them, so the rates describe those rows and the verdict changes run one way."""
    right = "; ".join(f"{NAMES[k][1] if k in NAMES else k}: {d['selection']['content_cut_right']} of "
                      f"{d['selection']['rows']} right under content" for k, d in dkf.items() if d.get("selection"))
    return ("These rows were chosen because the run cut them at the panel's cap (long, often runaway replies), so the "
            "rates describe those rows, not a random row, and the one-way verdict changes follow from that choice: a "
            "cut copy that runs to the cap is usually wrong (" + (right or "\u2013") + "), so a rerun can gain an answer "
            "and has little to lose. Under default kernels a rerun also differs from the run for reasons that have "
            "nothing to do with the cut, so the check does not separate the two")


def kernels(root: Path, key: str) -> str:
    if key in API:
        return "the provider's"
    p = root / f"results/ladder/scored_{key}.json"
    if not p.exists():
        return "–"
    rk = str(json.loads(p.read_text(encoding="utf-8")).get("run_key", ""))
    return "default (-dk fallback)" if rk.endswith("-dk") else "batch-invariant"


def tables_with_context(lines: list[tuple[int, str]]):
    """(line number, '###' heading, label line before the table, header, rows) of every table in a list of lines."""
    out, h3, label, i = [], "", "", 0
    while i < len(lines):
        no, ln = lines[i]
        if ln.startswith("### "):
            h3, label = ln[4:].strip(), ""
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-:| ]+\|$", lines[i + 1][1]):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i][1].startswith("|"):
                rows.append([c.strip() for c in lines[i][1].strip().strip("|").split("|")])
                i += 1
            out.append((no, h3, label, head, rows))
            continue
        if ln.strip():
            label = ln.strip()
        i += 1
    return out


def sections(text: str, marker: str = "## ") -> dict[str, list[tuple[int, str]]]:
    out, cur = {"": []}, ""
    for no, ln in enumerate(text.splitlines(), 1):
        if ln.startswith(marker) and not ln.startswith(marker + "#"):
            cur = ln[len(marker):].strip()
            out[cur] = []
            continue
        out[cur].append((no, ln))
    return out


def bullet_lines(name: str, lines: list[tuple[int, str]]) -> list[Line]:
    return [Line(ln[2:], f"{name}:{no}") for no, ln in lines if ln.startswith("- ")]


def check_text(T: Tally, F: Facts, ladder_md: Path, note_md: Path) -> int:
    """Part 2: every line and every table of ladder.md (sections 1 to 13 and the lines above them) and of
    ladder_note.md. Each line is masked for labels and references, its numbers are checked by the handler of its
    section, and any number left over is a mismatch; a table no checker reads is a mismatch."""
    n0 = T.n
    X = TextCheck(T, F)
    md = sections(ladder_md.read_text(encoding="utf-8"))
    sec = {k.split(".")[0]: v for k, v in md.items() if k[:1].isdigit()}
    T.text("ladder.md: numbered sections", [str(i) for i in range(1, 14)], sorted(sec, key=int))
    glance = [x for x in tables_with_context(sec.get("1", []))]
    if len(glance) != 2 and F.present:
        T.n += 1
        T.bad.append(f"ladder.md section 1: {len(glance)} glance tables, expected 2")
    for name, lines in [("", md.get("", []))] + sorted(sec.items(), key=lambda kv: int(kv[0])):
        for no, h3, label, head, rows in tables_with_context(lines):
            if not X.table(name, f"ladder.md:{no}", h3, label, head, rows) and \
                    any(re.search(r"\d", c) for r in rows for c in r):
                T.n += 1
                T.bad.append(f"ladder.md:{no}: a table with numbers that no checker reads: {head!r}")
        h3 = ""
        for no, t in lines:
            if not t.strip() or t.startswith("|"):
                continue
            if t.startswith("### "):
                h3 = t[4:]
            bullet = t.startswith("- ")
            keep = name in ("", "4", "9") or t.startswith("#")
            ln = Line(t if keep or not bullet else t[2:], f"ladder.md:{no}")
            ln.mask_structural()
            X.constants(ln)
            if name == "":
                X.preamble(ln)
            elif name == "1" and bullet:
                X.finding(ln)
            elif name == "2" and bullet:
                X.limits(ln)
            elif name == "3" and bullet:
                X.asks(ln)
            elif name == "4":
                X.sec4_line(ln)
            elif name == "5" and t.startswith("### "):
                X.family_heading(ln)
            elif name == "6" and t.startswith("Panel family members"):
                tm = re.match(TASK_RE, h3)
                if tm:
                    X.marks_line(ln, TASK_ROW[tm.group(1)])
                else:
                    X.fail(ln, "panel family members line outside a benchmark heading")
            elif name == "9":
                X.sec9_line(ln)
            elif name == "10" and bullet:
                X.footnote(ln)
            elif name == "11":
                X.sec11(ln)
            elif name == "12" and bullet:
                X.edge(ln)
            elif name == "13" and bullet:
                X.sources(ln)
            X.finish(ln)
    X.sec9_done()
    # the ladder note, all of it
    wn_text = note_md.read_text(encoding="utf-8")
    first = wn_text.splitlines()[0] if wn_text else ""
    m = re.match(r"^# Size ladder: note \(" + STAMP + r"\)$", first)
    T.text("ladder_note.md:1: created", F.J.get("created"), m.group(1) if m else None)
    wsec = sections(wn_text)
    if getattr(F, "orderings", None) is not None:
        T.text("ladder_note.md: the orderings section is present", True,
               any(n.startswith(ORD_SECTION) for n in wsec))
    for name, lines in wsec.items():
        tabs = tables_with_context(lines)
        for no, _, label, h, rows in tabs:
            if name.startswith("Headline numbers"):
                X.headline(f"ladder_note.md:{no}", h, rows)
            elif any(re.search(r"\d", c) for r in rows for c in r):
                T.n += 1
                T.bad.append(f"ladder_note.md:{no}: a table with numbers that no checker reads: {h!r}")
        if name == "":
            continue
        for no, t in lines:
            if not t.strip() or t.startswith("|") or t.startswith("#"):
                continue
            is_b = t.startswith("- ")
            ln = Line(t[2:] if is_b else t, f"ladder_note.md:{no}")
            ln.mask_structural()
            X.constants(ln)
            if name.startswith(ORD_SECTION):
                X.orderings(ln, is_b)
                X.finish(ln)
                continue
            if is_b:
                if name == "What is in":
                    X.what_is_in(ln)
                elif name.startswith("Family ladders") or name.startswith("Against the panel"):
                    X.finding(ln)
                elif name.startswith("Footnotes"):
                    X.footnote(ln)
                elif name == "Limits":
                    X.limits(ln)
                elif name == "What waits":
                    X.what_waits(ln)
            elif name == "What is in":
                X.size_order_note(ln)
            elif name.startswith("Family ladders") and t.startswith("Rungs, smallest first"):
                X.wn_family_para(ln)
            X.finish(ln)
    return T.n - n0


# ----------------------------------------------------------------------------------------------------------------

def run(root: Path, ladder_json: Path, nebius_md: Path | None, out=print, ladder_md: Path | None = None,
        note_md: Path | None = None) -> tuple[int, int]:
    J = json.loads(ladder_json.read_text(encoding="utf-8"))
    T = Tally()
    rec = {}
    for key in FILES:
        mj = g(J, "models", key)
        if mj is None:
            T.n += 1
            T.bad.append(f"{key}: absent from ladder.json (a model with no rows must still be listed)")
            rec[key] = {r: {} for r in READINGS}
            continue
        has_rows = any((root / FILES[key][r].format(task=t, key=key)).exists() for t in TASKS for r in READINGS)
        T.text(f"{key}/status_missing", str(not has_rows), str(mj.get("status") == "missing"))
        rec[key] = check_model(T, key, mj, root)
    panel = panel_tables(root)
    ladder_md = ladder_md or ladder_json.parent / "ladder.md"
    note_md = note_md or ladder_json.parent / "ladder_note.md"
    # The cut-against-uncut checks take their quantities from what ladder.md prints (its section 6, glance and section 5
    # tables), never from the analysis's code.
    printed = printed_lists(ladder_md.read_text(encoding="utf-8")) if ladder_md.exists() else None
    if not printed or not printed["placed"] or not printed["family"]:
        T.n += 1
        T.bad.append(f"{ladder_md}: no placed or family rows read, so the cut-against-uncut lists are empty")
    # the corrected gain's rule as RULE_TEXT states it (recompute follows it; a part it does not state is a mismatch)
    parts, problems = cg_rule(root)
    T.n += 1
    T.bad += problems
    out("Corrected gain, as src/baseline_mixture.RULE_TEXT states it (read here): w = share of requested outputs with "
        f"n_output_tokens {parts.get('compare', '?')} the forced cap, per configuration, mean over the "
        f"{parts.get('w_draws', '?')} draws; gain = CoT - forced over the {parts.get('gain_draws', '?')} draws; "
        f"{parts.get('form', '?')}")
    # the stop's unread replies, read here from the raw replies (ground truth, part 1)
    G = ground_truth_prepare(root, out)
    F = build_facts(root, J, rec, panel, printed, G["stop"] if G else None)
    F.gt = G
    # the orderings, when scripts/ladder_orderings.py has written its file
    oj = ladder_json.parent / ORD_JSON
    F.orderings = F.ord_json = None
    n_ord = None
    if oj.exists():
        F.ord_json = json.loads(oj.read_text(encoding="utf-8"))
        F.orderings = orderings_facts(root, rec, panel)
        n_ord = check_orderings(T, F.orderings, F.ord_json, root)
    n_json0 = T.n
    check_json_extras(T, F)
    n_json = T.n - n_json0
    n_md = 0
    ds_all = rec.get(DS, {})
    if nebius_md is not None and nebius_md.exists() and ds_all and all(len(v) == 4 for v in ds_all.values()):
        n_md = check_nebius_md(T, ds_all, nebius_md)
    n_txt = 0
    if ladder_md.exists() and note_md.exists():
        n_txt = check_text(T, F, ladder_md, note_md)
    else:
        T.n += 1
        T.bad.append(f"{ladder_md} or {note_md} not found; the printed text was not checked")
    gt_words = ground_truth_stage(T, F, out)
    for b in T.bad:
        out(f"MISMATCH {b}")
    out(f"Families, placements, gaps, cut against uncut, completeness: {n_json} values compared")
    out(f"DeepSeek against {nebius_md.relative_to(root) if nebius_md else '-'}: {n_md} printed cells compared")
    out(f"Printed text (ladder.md, every line and table of sections 1 to 13 and above them, template SDs "
        f"recomputed; ladder_note.md, all of it): {n_txt} values compared")
    ord_words = ""
    if n_ord is not None:
        out(f"Orderings ({ORD_JSON}, the lad and comb macros of {NUMBERS_TEX}; recounted from the scored rows, own "
            f"bootstrap of {ORD_BOOT:,} resamples, seed {ORD_SEED}, bounds within {ORD_BOOT_TOL}): {n_ord} values compared"
            + ("" if F.orderings.get("qwen") else "; Qwen3.5-122B-A10B's reply forms not read (no raw replies)"))
        ord_words = f"; orderings recounted ({n_ord} values)"
    out(f"VERIFY: {T.n} values checked, {len(T.bad)} mismatches; {gt_words}{ord_words}")
    return T.n, len(T.bad)


def ground_truth_prepare(root: Path, out=print) -> dict | None:
    """Ground truth, part 1 (GT.stop_unread): the forced replies the stop left unread, per ladder model and family rung,
    read here from the raw replies with the rule as src/forced_stop.py states it. None when the raw replies are not
    under root (the tests' synthetic roots); the VERIFY line then says the stage did not run."""
    if not GT.available(root):
        out(f"GROUND TRUTH: not run (no raw replies under {root})")
        return None
    rule = GT.StopRule(root)
    keys = [(k, "ladder") for k in FILES] + [(k, "panel") for f in FAMILIES.values() for k, s in f["rungs"]
                                             if s == "panel"]
    reader = GT.Reader(root, rule.wrappers)
    su = GT.stop_unread(root, rule, keys, reader)      # ranges the reader reads are not unread
    stop = {}
    for k, v in su.items():
        for t, x in v.items():
            disc, _ = sampler_draws(t, root)
            zs = x["arms"].get("forced_direct", {})
            stop.setdefault(k, {})[t] = {"rows": int(zs.get("rows", 0)), "unread": int(zs.get("unread", 0)),
                                         "unfinished": int(zs.get("unfinished", 0)),
                                         "discovery_pp": GT.discovery_pp(x["by_config"], disc)}
    return {"rule": rule, "unread": su, "stop": stop, "reader": reader}


def ground_truth_stage(T: Tally, F: Facts, out=print) -> str:
    """Ground truth, part 2: the analysis's stop counts against the raw replies read here; the API models' scored
    forced replies against the box they closed in their whole reply; a seeded sample of raw replies read here against
    the scored rows. Returns the words for the VERIFY line."""
    G, J = getattr(F, "gt", None), F.J
    if not G:
        return "ground truth not run (no raw replies)"
    su, rule = G["unread"], G["rule"]
    # (a) the stop's counts: ladder.json stop_counts (audit) and stop_unread (per rung) against the raw replies
    names = {"rows": "rows", "open_box": "open", "completed": "completed", "open_inner_box": "inner",
             "open_range": "range", "open_range_read": "range_read", "open_range_unread": "range_unread",
             "open_unfinished": "unfinished", "open_other": "other", "unread": "unread",
             "open_inner_box_restart": "restart"}
    au = (J.get("stop_counts") or {}).get("audit") or {}
    T.text("ground truth: rule version in ladder.json stop_counts", rule.version,
           (J.get("stop_counts") or {}).get("version"))
    for k in FILES:
        for t in STOP_TASKS:
            for arm in GT.FORCED_ARMS:
                mine = su.get(k, {}).get(t, {}).get("arms", {}).get(arm)
                if mine is None:
                    continue
                theirs = au.get(k, {}).get(t, {}).get(arm, {})
                for a, b in names.items():
                    T.num(f"ground truth: stop_counts {k} {t} {arm} {a}", int(mine.get(b, 0)), theirs.get(a))
    sj = J.get("stop_unread") or {}
    for k, v in G["stop"].items():
        for t, x in v.items():
            y = sj.get(k, {}).get(t, {})
            for f_ in ("rows", "unread", "unfinished", "discovery_pp"):
                T.num(f"ground truth: stop_unread {k} {t} {f_}", x[f_], y.get(f_))
    # (b) the API models' full replies
    fr = GT.full_reply_check(F.root, rule, G["reader"])
    tot = Counter()
    for k, d in fr.items():
        c = d["counts"]
        tot.update({x: c.get(x, 0) for x in ("compared", "agree", "false_gain", "false_loss")})
        out(f"GROUND TRUTH full replies {k}: {c.get('replies', 0)} forced replies the cut leaves open, "
            f"{c.get('full_not_closed', 0)} never closed in the whole reply, {c.get('compared', 0)} compared, "
            f"{c.get('agree', 0)} agree, {c.get('false_gain', 0)} scored correct where the whole box is not, "
            f"{c.get('false_loss', 0)} scored not correct where the whole box is "
            f"({c.get('unread_false_loss', 0)} of them left unread by the rule)")
        for e in d["examples"]:
            out(f"GROUND TRUTH full replies {k} example: {e}")
        # gated apart. The replies the readers read (completed, inner box, ranges read): at most
        # FULL_MAX_DISAGREE of them may disagree with the whole reply. The replies left unread: none may be scored
        # correct (a false gain), and their false losses must be the stop's loss that ladder.json reports (F1)
        T.n += 1
        rc, rd = c.get("read_compared", 0), c.get("read_disagree", 0)
        out(f"GROUND TRUTH full replies {k}, read: {rc} compared, {rd} disagree; left unread: "
            f"{c.get('unread_compared', 0)} compared, {c.get('unread_false_gain', 0)} false gains, "
            f"{c.get('unread_false_loss', 0)} false losses")
        if rc and rd > GT.FULL_MAX_DISAGREE * rc:
            T.bad.append(f"ground truth: {k}: {rd} of {rc} scored forced replies the readers read disagree with the "
                         f"whole reply (more than {float(GT.FULL_MAX_DISAGREE):.0%})")
        # each read class on its own (completed, inner box, range): a class of fewer than 1/FULL_MAX_DISAGREE
        # replies may not disagree at all, so a small class read wrongly throughout fails here
        for st in sorted(x[len("read_compared_"):] for x in c if x.startswith("read_compared_")):
            T.n += 1
            n_st, d_st = c.get(f"read_compared_{st}", 0), c.get(f"read_disagree_{st}", 0)
            out(f"GROUND TRUTH full replies {k}, read class {st}: {n_st} compared, {d_st} disagree")
            if n_st and d_st > GT.FULL_MAX_DISAGREE * n_st:
                T.bad.append(f"ground truth: {k}: read class {st}: {d_st} of {n_st} scored forced replies disagree "
                             f"with the whole reply (more than {float(GT.FULL_MAX_DISAGREE):.0%})")
        T.num(f"ground truth: {k} unread forced replies scored correct where the whole box is not",
              int(c.get("unread_false_gain", 0)), 0)
        doc = ((J.get("stop_counts") or {}).get("full_replies") or {}).get(k, {})
        T.num(f"ground truth: {k} unread replies correct in the whole box (ladder.json stop_counts)",
              int(c.get("unread_false_loss", 0)), doc.get("unread_false_loss"))
    # (c) a seeded sample of raw replies read here, every benchmark, both readings, every arm: every
    # disagreement is a mismatch; each cell's one-sided 95% upper bound on its share of disagreements is printed.
    # the main panel's replies are sampled too (panel_sample_check), every model, benchmark and arm
    sc = GT.sample_check(F.root, rule, G["reader"], list(FILES))
    ps = GT.panel_sample_check(F.root, rule, G["reader"])
    p_agree = sum(a for a, _ in ps["cells"].values())
    p_n = sum(b for _, b in ps["cells"].values())
    for (k, t, rd, arm, st), (a, b) in sorted(ps["cells"].items()):
        T.n += 1
        out(f"GROUND TRUTH panel sample cell {k} {t} {arm}: {a} of {b} read alike, {b - a} disagree")
    for (k, t, rd, arm, st), a, b, ub in GT.cell_failures(ps["cells"]):
        T.bad.append(f"ground truth: panel sample {k} {t} {arm}: {b - a} of {b} read differently here and in the "
                     f"scored rows")
    for d in ps["disagree"]:
        out(f"GROUND TRUTH panel sample disagreement: {d}")
    out(f"GROUND TRUTH panel sample: {p_agree} of {p_n} main-panel replies read alike here and in the scored rows, "
        f"{len(ps['cells'])} cells ({GT.PANEL_SAMPLE_N} per model, benchmark and arm; seed {GT.PANEL_SEED}; GSM8K, "
        f"MATH-500, BIG-Bench Hard, HotpotQA and the five-shot forced arms)")
    agree = sum(a for a, _ in sc["cells"].values())
    n = sum(b for _, b in sc["cells"].values())
    for (k, t, rd, arm, st), (a, b) in sorted(sc["cells"].items()):
        T.n += 1
        ub = GT.upper_bound(b - a, b)
        out(f"GROUND TRUTH sample cell {k} {t} {rd} {arm} ({st}): {a} of {b} read alike, {b - a} disagree "
            f"(upper {GT.BOUND_LEVEL:.0%} bound on the share {ub:.4f})" if b else
            f"GROUND TRUTH sample cell {k} {t} {rd} {arm} ({st}): no rows")
    for (k, t, rd, arm, st), a, b, ub in GT.cell_failures(sc["cells"]):
        T.bad.append(f"ground truth: sample {k} {t} {rd} {arm} ({st}): {b - a} of {b} read differently here and in "
                     f"the scored rows (more than {GT.SAMPLE_MAX_DISAGREE}; upper {GT.BOUND_LEVEL:.0%} bound on the "
                     f"share {ub:.4f})")
    ub_all = GT.upper_bound(n - agree, n)
    cells_n = [b for a, b in sc["cells"].values() if b]
    out(f"GROUND TRUTH sample: {agree} of {n} replies read alike here and in the scored rows, {len(sc['cells'])} cells "
        f"({GT.SAMPLE_N} per model, benchmark, reading and arm, and up to {GT.SAMPLE_CUT_N} replies cut at the panel's "
        f"cap per arm in the cut reading; seed {GT.SAMPLE_SEED}; GSM8K, MATH-500, BIG-Bench Hard and HotpotQA; uncut "
        f"and cut). Upper {GT.BOUND_LEVEL:.0%} bound on the share of disagreements over all cells: {ub_all:.5f}. A cell "
        f"of {GT.SAMPLE_N} fails at its first disagreement, so a reading error in 1% of a cell's rows is caught with "
        f"probability {1 - 0.99 ** GT.SAMPLE_N:.2f}, in 2% with {1 - 0.98 ** GT.SAMPLE_N:.2f}"
        + (f" (smallest cell {min(cells_n)} replies)" if cells_n else ""))
    for d in sc["disagree"]:
        out(f"GROUND TRUTH sample disagreement: {d}")
    return (f"ground truth: sample {agree} of {n} read alike (4 benchmarks, both readings); panel sample {p_agree} "
            f"of {p_n} read alike; API full replies "
            f"{tot['agree']} of {tot['compared']} alike ({tot['false_gain']} false gains, {tot['false_loss']} false "
            f"losses)")


# ----------------------------------------------------------------------------------------------------------------
# Orderings on the size ladder (results/ladder/analysis/ladder_orderings.json), recounted
# here from the scored rows (the six models' exact means of `rec`) and the panel's result files (`panel_tables`, and
# HotpotQA content's discovery-draw means from results/hotpot_content/hotpot_content.json), with this checker's own
# pair loop, own resampling (own seed) and own reading of the raw replies.
# ----------------------------------------------------------------------------------------------------------------

def own_pairs(keys: list[str], g1: list, g2: list) -> dict:
    """Pairs i < j: reversed when g1 and g2 order them oppositely; a pair with |difference| <= TOL under either is a
    tie, neither reversed nor kept, and stays among the pairs."""
    rev, tie, gaps = [], [], {}
    above = {th: [0, 0] for th in ORD_GAPS}         # pairs whose two gaps both exceed th, and the reversed among them
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d1, d2 = g1[i] - g1[j], g2[i] - g2[j]
            small = min(abs(d1), abs(d2))
            is_rev = False
            if abs(d1) <= TOL or abs(d2) <= TOL:
                tie.append((keys[i], keys[j]))
            elif (d1 > 0) != (d2 > 0):
                rev.append((keys[i], keys[j]))
                gaps[(keys[i], keys[j])] = (abs(d1), abs(d2))
                is_rev = True
            for th in ORD_GAPS:
                if small > th:
                    above[th][0] += 1
                    above[th][1] += is_rev
    n = len(keys)
    return {"reversed": len(rev), "pairs": n * (n - 1) // 2, "ties": len(tie), "rev_set": set(rev), "models": n,
            "above": {th: {"pairs": a, "reversed": r} for th, (a, r) in above.items()}, "gaps": gaps}


def own_pool(blocks: list[dict]) -> dict:
    return {"reversed": sum(b["reversed"] for b in blocks), "pairs": sum(b["pairs"] for b in blocks),
            "ties": sum(b["ties"] for b in blocks),
            "above": {th: {"pairs": sum(b["above"][th]["pairs"] for b in blocks),
                           "reversed": sum(b["above"][th]["reversed"] for b in blocks)} for th in ORD_GAPS}}


def own_boot(groups: list[tuple], n_boot: int, rng) -> list[float]:
    """Percentile 95% interval of the reversed share under resampling models with replacement (one draw of the union
    of the groups' models serves every group; a pair of a model with its own copy is no pair; a model absent from a
    group adds nothing there)."""
    import numpy as np
    union: list[str] = []
    for keys, _, _ in groups:
        union += [k for k in keys if k not in union]
    mats = []
    for keys, g1, g2 in groups:
        n = len(keys)
        R = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                d1, d2 = g1[i] - g1[j], g2[i] - g2[j]
                R[i, j] = abs(d1) > TOL and abs(d2) > TOL and (d1 > 0) != (d2 > 0)
        pos = np.array([keys.index(k) if k in keys else -1 for k in union])
        mats.append((R, pos))
    U = len(union)
    shares = []
    for _ in range(n_boot):
        ix = rng.integers(0, U, U)
        rv = n_ = 0
        for R, pos in mats:
            loc = pos[ix]
            loc = loc[loc >= 0]
            a, b = np.triu_indices(len(loc), 1)
            ok = loc[a] != loc[b]
            rv += int(R[loc[a][ok], loc[b][ok]].sum())
            n_ += int(ok.sum())
        if n_:
            shares.append(rv / n_)
    v = np.sort(np.array(shares))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def hand_counts_from_file(root: Path) -> dict:
    """The hand counts (HAND_COUNTS): reversed pairs of 15 per benchmark, counted by hand."""
    return dict(HAND_COUNTS)


def panel_gains(root: Path, panel: dict, t: str, s: str) -> tuple[list, list, list]:
    """The panel's model keys and its CoT - requested and CoT - forced on the paper's basis: panel_tables for GSM8K,
    MATH-500, BIG-Bench Hard and HotpotQA strict (discovery draws); HotpotQA content on the discovery draws from
    results/hotpot_content/hotpot_content.json (arm_effect, the basis of \\aeHqReversedContent)."""
    if t == "hotpot" and s == "content":
        disc, _ = sampler_draws("hotpot", root)
        B = len(disc) * PANEL_ITEMS["hotpot"]
        H = json.loads((root / "results/hotpot_content/hotpot_content.json").read_text(encoding="utf-8"))
        pm = H["arm_effect"]["content"]["per_model"]
        keys = list(pm)
        ex = {k: {a: exact_of(pm[k][a], B) for a in ("requested", "forced", "cot")} for k in keys}
        return keys, [ex[k]["cot"] - ex[k]["requested"] for k in keys], [ex[k]["cot"] - ex[k]["forced"] for k in keys]
    df = panel[t]
    keys = list(df.index)
    return keys, [df.at[k, f"cot_minus_requested_{s}"] for k in keys], [df.at[k, f"cot_minus_forced_{s}"] for k in keys]


def orderings_facts(root: Path, rec: dict, panel: dict) -> dict:
    """Everything of the orderings, recounted here."""
    import numpy as np
    O: dict = {"six": {}, "big": {}, "comb": {}, "panel": {}, "boot": {}}
    keys6 = list(FILES)
    rng = np.random.default_rng(ORD_SEED)
    for r in READINGS:
        for s in SCORE_COL:
            for basis in ("discovery", "all_draws"):
                blk = {}
                for t in TASKS:
                    g1 = [rec[k][r][t][s][basis]["cot_minus_requested"] for k in keys6]
                    g2 = [rec[k][r][t][s][basis]["cot_minus_forced"] for k in keys6]
                    blk[t] = own_pairs(keys6, g1, g2)
                blk["pooled"] = own_pool([blk[t] for t in DISC_TASKS])
                O["six"].setdefault(r, {}).setdefault(s, {})[basis] = blk
            bb = {}
            for t in TASKS:
                bb[t] = own_pairs(BIG, [rec[k][r][t][s]["discovery"]["cot_minus_requested"] for k in BIG],
                                  [rec[k][r][t][s]["discovery"]["cot_minus_forced"] for k in BIG])
            bb["pooled"] = own_pool([bb[t] for t in DISC_TASKS])
            O["big"].setdefault(r, {})[s] = bb
            cb, groups = {}, []
            for t in TASKS:
                pk, p1, p2 = panel_gains(root, panel, t, s)
                keys = pk + keys6
                g1 = p1 + [rec[k][r][t][s]["discovery"]["cot_minus_requested"] for k in keys6]
                g2 = p2 + [rec[k][r][t][s]["discovery"]["cot_minus_forced"] for k in keys6]
                cb[t] = own_pairs(keys, g1, g2)
                cb[t]["ci_own"] = own_boot([(keys, g1, g2)], ORD_BOOT, rng)
                if t in DISC_TASKS:
                    groups.append((keys, g1, g2))
            cb["pooled"] = own_pool([cb[t] for t in DISC_TASKS])
            cb["pooled"]["models"] = len({k for keys, _, _ in groups for k in keys})
            cb["pooled"]["ci_own"] = own_boot(groups, ORD_BOOT, rng)
            O["comb"].setdefault(r, {})[s] = cb
    for s in SCORE_COL:
        pb, groups = {}, []
        for t in TASKS:
            keys, g1, g2 = panel_gains(root, panel, t, s)
            pb[t] = own_pairs(keys, g1, g2)
            pb[t]["ci_own"] = own_boot([(keys, g1, g2)], ORD_BOOT, rng)
            if t in DISC_TASKS:
                groups.append((keys, g1, g2))
        pb["pooled"] = own_pool([pb[t] for t in DISC_TASKS])
        pb["pooled"]["models"] = len({k for keys, _, _ in groups for k in keys})
        pb["pooled"]["ci_own"] = own_boot(groups, ORD_BOOT, rng)
        O["panel"][s] = pb
    # the hand counts and the pairs whose status differs between the 48 draws and the discovery draws
    O["hand"] = {"file": hand_counts_from_file(root), "diff": {}}
    for t in TASKS:
        a = O["six"]["cut"]["content"]["all_draws"][t]["rev_set"]
        d = O["six"]["cut"]["content"]["discovery"][t]["rev_set"]
        O["hand"]["diff"][t] = sorted(a ^ d)
    # w by family and size, and the panel's range
    O["w"] = {"ladder": {k: {t: rec[k]["cut"][t]["w"]["discovery"] for t in TASKS} for k in keys6},
              "same": all(rec[k]["cut"][t]["w"]["discovery"] == rec[k]["uncut"][t]["w"]["discovery"]
                          for k in keys6 for t in TASKS), "panel": {}, "fam": {}}
    for t in TASKS:
        vals = sorted((panel[t].at[k, "w"] for k in panel[t].index), key=float)
        n = len(vals)
        med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        O["w"]["panel"][t] = {"min": vals[0], "max": vals[-1], "median": med}
    for fam, f in FAMILIES.items():
        O["w"]["fam"][fam] = {}
        for t in TASKS:
            vals = [rec[k]["cut"][t]["w"]["discovery"] if src == "ladder" else panel[t].at[k, "w"]
                    for k, src in f["rungs"]]
            O["w"]["fam"][fam][t] = {"values": vals, "direction": direction(vals, "w"),
                                     "keys": [k for k, _ in f["rungs"]]}
    # the shift and its parts, both readings and scores
    O["shift"] = {r: {s: {k: {t: rec[k][r][t]["split16"][s] for t in TASKS} for k in keys6} for s in SCORE_COL}
                  for r in READINGS}
    O["long_share"] = {r: {k: {t: rec[k][r][t]["split16"]["long_share"] for t in TASKS} for k in keys6}
                       for r in READINGS}
    B = json.loads((root / "results/baseline_mixture/baseline_mixture.json").read_text(encoding="utf-8"))
    H = json.loads((root / "results/hotpot/decomposition_thresholds_post_hoc.json").read_text(encoding="utf-8"))
    O["shift_panel"] = {}
    for s in SCORE_COL:
        for t in TASKS:
            pm = (H["variants"]["cap16"][s]["sets"]["discovery"]["per_model"] if t == "hotpot"
                  else B["decomposition"][t]["sets"]["discovery"][s]["per_model"])
            bd = len(sampler_draws(t, root)[0]) * PANEL_ITEMS[t]       # each value is 100 m / (draws x items)
            O["shift_panel"].setdefault(s, {})[t] = {
                q: (min((exact_of(x, bd) for x in pm[q] if x is not None and math.isfinite(x)), key=float),
                    max((exact_of(x, bd) for x in pm[q] if x is not None and math.isfinite(x)), key=float))
                for q in ("shift", "part_long", "part_short")}
    O["qwen"] = qwen_unreadable(root)
    return O


def qwen_unreadable(root: Path) -> dict | None:
    """Qwen3.5-122B-A10B's zero-shot requested replies the strict scorer flags unreadable, read here: per benchmark the
    rows, the unreadable ones, those with no box at all (no \\boxed in the raw reply) by form (an option letter in
    parentheses; digits and signs once LaTeX commands are set aside; anything else), the boxed ones, the median token
    count of those with no box, and how many of those the content reading counts correct. None without raw replies."""
    if not GT.available(root):
        return None
    run_ = (root / "data/d2_out/ladder2/chosen" / QWEN).read_text(encoding="utf-8").strip()
    out = {}
    for t in TASKS:
        sc = pd.read_parquet(root / FILES[QWEN]["cut"].format(task=t, key=QWEN),
                             columns=["request_id", "arm", "compliant", "content_correct", "n_output_tokens"])
        sc = sc[sc["arm"] == "direct_answer"]
        raw = GT.shard_rows(root / "data/d2_out/ladder2" / run_ / t, ["direct_answer"],
                            cols=("request_id", "arm", "output_text"))
        text = dict(zip(raw["request_id"], raw["output_text"]))
        forms = Counter()
        tokens, correct = [], 0
        unread = 0
        for rid, ok, cc, nt in zip(sc["request_id"], sc["compliant"], sc["content_correct"], sc["n_output_tokens"]):
            if bool(ok):
                continue
            unread += 1
            x = text[rid]
            if "\\boxed" in x:
                forms["box"] += 1
                continue
            y = x.strip()
            core = re.sub(r"\\[A-Za-z]+", "", y)
            if re.match(r"\([A-Z]\)", y):
                forms["bare option"] += 1
            elif re.search(r"\d", core) and re.fullmatch(r"[0-9\s.,+\-*/^()=%$:{}\[\]\\]+", core):
                forms["bare number"] += 1
            else:
                forms["bare words"] += 1
            tokens.append(int(nt))
            correct += int(cc)
        tokens.sort()
        n = len(tokens)
        med = (Fraction(tokens[n // 2]) if n % 2 else Fraction(tokens[n // 2 - 1] + tokens[n // 2], 2)) if n else None
        out[t] = {"rows": int(len(sc)), "unreadable": unread, "forms": dict(forms), "bare": n, "median": med,
                  "bare_correct": correct, "unread_correct": int(sc.loc[~sc["compliant"].astype(bool),
                                                                        "content_correct"].sum())}
    return out


def ord_fmt(v, d: int = 0) -> str:
    """A macro value as this checker prints it: Python's half-even formatting of the exact value (format(Fraction,
    '.<d>f')), a value that rounds to zero without its sign, minus as $-$, thousands of a whole number as {,}."""
    if v is None:
        return "--"
    if d == 0 and isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    s = format(Fraction(v), f".{d}f")
    if s.startswith("-") and set(s[1:]) <= set("0."):
        s = s[1:]
    return s.replace("-", "$-$")


def ord_expected_macros(O: dict, J: dict) -> dict:
    """The macros lad... and comb... as this checker computes them (bootstrap bounds: the file's, once its interval is
    found within ORD_BOOT_TOL of this checker's own)."""
    m: dict = {}
    for (s, r), V in ORD_VARIANT.items():
        for t in TASKS + ["pooled"]:
            tag = ORD_T[t]
            x, b, c = O["six"][r][s]["discovery"][t], O["big"][r][s][t], O["comb"][r][s][t]
            m[f"ladRev{tag}{V}"], m[f"ladPairs{tag}{V}"] = ord_fmt(x["reversed"]), ord_fmt(x["pairs"])
            m[f"ladRevPct{tag}{V}"] = ord_fmt(Fraction(100 * x["reversed"], x["pairs"]))
            m[f"ladTies{tag}{V}"] = ord_fmt(x["ties"])
            m[f"ladBigRev{tag}{V}"], m[f"ladBigPairs{tag}{V}"] = ord_fmt(b["reversed"]), ord_fmt(b["pairs"])
            m[f"ladBigTies{tag}{V}"] = ord_fmt(b["ties"])
            m[f"combRev{tag}{V}"], m[f"combPairs{tag}{V}"] = ord_fmt(c["reversed"]), ord_fmt(c["pairs"])
            m[f"combRevPct{tag}{V}"] = ord_fmt(Fraction(100 * c["reversed"], c["pairs"]))
            m[f"combRevShare{tag}{V}"] = ord_fmt(Fraction(c["reversed"], c["pairs"]), 3)
            m[f"combTies{tag}{V}"] = ord_fmt(c["ties"])
            m[f"combM{tag}{V}"] = ord_fmt(c["models"])
            ci = g(J, "reversals", "combined", r, s, t, "model_bootstrap", "ci95") or [None, None]
            m[f"combRevBootLo{tag}{V}"], m[f"combRevBootHi{tag}{V}"] = ord_fmt(ci[0], 2), ord_fmt(ci[1], 2)
            for th in ORD_GAPS:                       # the threshold rows
                w_ = ORD_GAP_WORD[th]
                for stem, blk in (("ladGap", x), ("ladBigGap", b), ("combGap", c)):
                    m[f"{stem}{w_}Pairs{tag}{V}"] = ord_fmt(blk["above"][th]["pairs"])
                    m[f"{stem}{w_}Rev{tag}{V}"] = ord_fmt(blk["above"][th]["reversed"])
    for s in SCORE_COL:
        V = ORD_VARIANT[(s, "cut")]
        for t in TASKS + ["pooled"]:
            tag = ORD_T[t]
            p = O["panel"][s][t]
            m[f"combPanelRev{tag}{V}"], m[f"combPanelPairs{tag}{V}"] = ord_fmt(p["reversed"]), ord_fmt(p["pairs"])
            m[f"combPanelRevPct{tag}{V}"] = ord_fmt(Fraction(100 * p["reversed"], p["pairs"]))
            m[f"combPanelRevShare{tag}{V}"] = ord_fmt(Fraction(p["reversed"], p["pairs"]), 3)
            m[f"combPanelM{tag}{V}"] = ord_fmt(p["models"])
            ci = g(J, "reversals", "panel", s, t, "model_bootstrap", "ci95") or [None, None]
            m[f"combPanelRevBootLo{tag}{V}"], m[f"combPanelRevBootHi{tag}{V}"] = ord_fmt(ci[0], 2), ord_fmt(ci[1], 2)
    W = O["w"]
    for t in TASKS:
        tag = ORD_T[t]
        for k, v in W["ladder"].items():
            m[f"ladW{ORD_MODEL[k]}{tag}"] = ord_fmt(v[t], 3)
        for fam, f in W["fam"].items():
            for k, v in zip(f[t]["keys"], f[t]["values"]):
                m[f"ladW{ORD_MODEL[k]}{tag}"] = ord_fmt(v, 3)
            m[f"ladWDir{ORD_FAM[fam]}{tag}"] = f[t]["direction"]
        for q, name in (("min", "Min"), ("max", "Max"), ("median", "Median")):
            m[f"ladWPanel{name}{tag}"] = ord_fmt(W["panel"][t][q], 3)
    for (s, r), V in ORD_VARIANT.items():
        for k in FILES:
            for t in TASKS:
                x = O["shift"][r][s][k][t]
                stem = f"{ORD_MODEL[k]}{ORD_T[t]}{V}"
                m[f"ladShift{stem}"] = ord_fmt(x["shift"], 1)
                m[f"ladShiftLong{stem}"] = ord_fmt(x["long"], 1)
                m[f"ladShiftShort{stem}"] = ord_fmt(x["short"], 1)
                if s == "content":
                    m[f"ladShiftLongShare{stem}"] = ord_fmt(O["long_share"][r][k][t], 3)
    for s in SCORE_COL:
        V = ORD_VARIANT[(s, "cut")]
        for t in TASKS:
            for q, qn in (("shift", ""), ("part_long", "Long"), ("part_short", "Short")):
                lo, hi = O["shift_panel"][s][t][q]
                m[f"ladShift{qn}PanelMin{ORD_T[t]}{V}"] = ord_fmt(lo, 1)
                m[f"ladShift{qn}PanelMax{ORD_T[t]}{V}"] = ord_fmt(hi, 1)
    if O["qwen"]:
        for t, x in O["qwen"].items():
            tag = ORD_T[t]
            m[f"ladQwenReqRows{tag}"], m[f"ladQwenReqUnread{tag}"] = ord_fmt(x["rows"]), ord_fmt(x["unreadable"])
            m[f"ladQwenReqUnreadPct{tag}"] = ord_fmt(Fraction(100 * x["unreadable"], x["rows"]), 1)
            m[f"ladQwenReqBare{tag}"] = ord_fmt(x["bare"])
            m[f"ladQwenReqBarePct{tag}"] = ord_fmt(Fraction(100 * x["bare"], x["unreadable"]), 1)
            m[f"ladQwenReqBareTokens{tag}"] = ord_fmt(x["median"], 1)
            m[f"ladQwenReqUnreadContentPct{tag}"] = ord_fmt(Fraction(100 * x["unread_correct"], x["unreadable"]), 1)
    return m


def check_orderings(T: Tally, O: dict, J: dict, root: Path) -> int:
    """ladder_orderings.json and the macros lad... and comb... of paper/numbers.tex against this checker's own counts."""
    n0 = T.n
    pth = "ladder_orderings.json"
    T.num(f"{pth}: n_boot", ORD_BOOT, J.get("n_boot"))
    T.num(f"{pth}: split_tokens", SPLIT_TOKENS, J.get("split_tokens"))
    RV = J.get("reversals") or {}

    def cmp(where, mine, theirs, names=True):
        for f_ in ("reversed", "pairs", "ties"):
            T.num(f"{pth}: {where} {f_}", mine[f_], g(theirs, f_))
        if names and "rev_set" in mine:
            got = {tuple(p) for p in (g(theirs, "reversed_pairs") or [])}
            T.text(f"{pth}: {where} reversed pairs", sorted(mine["rev_set"]), sorted(got))
        # the threshold rows
        rows = {float(x.get("min_gap_pp")): x for x in (g(theirs, "by_threshold") or [])}
        for th in ORD_GAPS:
            x = rows.get(float(th)) or {}
            T.num(f"{pth}: {where} both gaps above {th} pp, pairs", mine["above"][th]["pairs"], x.get("pairs_both_above"))
            T.num(f"{pth}: {where} both gaps above {th} pp, reversed", mine["above"][th]["reversed"], x.get("reversed"))
        if names and "gaps" in mine and not where.startswith("combined"):      # the six's exact gaps (printed)
            got = {tuple(x["pair"]): x for x in (g(theirs, "reversed_gaps") or [])}
            T.text(f"{pth}: {where} pairs with gaps", sorted(mine["gaps"]), sorted(got))
            for p_, (d1, d2) in sorted(mine["gaps"].items()):
                pr = (got.get(p_) or {}).get("printed") or [None, None]
                T.printed(f"{pth}: {where} {p_} gap CoT - requested", pr[0], d1)
                T.printed(f"{pth}: {where} {p_} gap CoT - forced", pr[1], d2)

    def ci_check(where, mine, theirs):
        ci = g(theirs, "model_bootstrap", "ci95")
        T.n += 1
        if not ci or any(abs(a - b) > ORD_BOOT_TOL for a, b in zip(mine, ci)):
            T.bad.append(f"{pth}: {where} model bootstrap {ci} against this checker's {mine} (own seed; tolerance "
                         f"{ORD_BOOT_TOL})")

    for r in READINGS:
        for s in SCORE_COL:
            for basis in ("discovery", "all_draws"):
                for t in TASKS + ["pooled"]:
                    cmp(f"six {r} {s} {basis} {t}", O["six"][r][s][basis][t], g(RV, "six", r, s, basis, t))
            for t in TASKS + ["pooled"]:
                cmp(f"70B and above {r} {s} {t}", O["big"][r][s][t], g(RV, "big", r, s, t))
                cmp(f"combined {r} {s} {t}", O["comb"][r][s][t], g(RV, "combined", r, s, t))
                T.num(f"{pth}: combined {r} {s} {t} models", O["comb"][r][s][t]["models"],
                      g(RV, "combined", r, s, t, "union" if t == "pooled" else "models"))
                ci_check(f"combined {r} {s} {t}", O["comb"][r][s][t]["ci_own"], g(RV, "combined", r, s, t))
    for s in SCORE_COL:
        for t in TASKS + ["pooled"]:
            cmp(f"panel {s} {t}", O["panel"][s][t], g(RV, "panel", s, t), names=False)
            ci_check(f"panel {s} {t}", O["panel"][s][t]["ci_own"], g(RV, "panel", s, t))
    # the hand counts
    hc = J.get("hand_check") or {}
    for t in TASKS:
        w = O["hand"]["file"].get(t)
        T.num(f"{pth}: hand count {t} (from the file)", w[0] if w else None, g(hc, t, "hand"))
        T.num(f"{pth}: hand count {t}, 48 draws", O["six"]["cut"]["content"]["all_draws"][t]["reversed"],
              g(hc, t, "all_draws"))
        T.num(f"{pth}: hand count {t}, discovery", O["six"]["cut"]["content"]["discovery"][t]["reversed"],
              g(hc, t, "discovery"))
        got = sorted(tuple(d["pair"]) for d in (g(hc, t, "pairs_differing_48_vs_discovery") or []))
        T.text(f"{pth}: hand count {t}, pairs differing", O["hand"]["diff"][t], got)
        # where the hand count and the 48-draw count differ, the 48-draw reversed pairs are named
        a48 = O["six"]["cut"]["content"]["all_draws"][t]
        want = sorted(a48["gaps"]) if w and w[0] != a48["reversed"] else []
        T.text(f"{pth}: hand count {t}, 48-draw pairs named", want,
               sorted(tuple(d["pair"]) for d in (g(hc, t, "all_draws_reversed_gaps") or [])))
    # w
    for k, v in O["w"]["ladder"].items():
        for t in TASKS:
            T.num(f"{pth}: w {k} {t}", v[t], g(J, "w", "ladder", k, t, "w"))
    for t in TASKS:
        for q in ("min", "max", "median"):
            T.num(f"{pth}: w panel {q} {t}", O["w"]["panel"][t][q], g(J, "w", "panel", t, q))
    for fam, f in O["w"]["fam"].items():
        for t in TASKS:
            T.text(f"{pth}: w direction {fam} {t}", f[t]["direction"], g(J, "w", "families", fam, "tasks", t, "direction"))
            for i, v in enumerate(f[t]["values"]):
                T.num(f"{pth}: w {fam} {t} rung {i}", v, (g(J, "w", "families", fam, "tasks", t, "values") or [None] * 9)[i])
    T.text(f"{pth}: w same in both readings", O["w"]["same"], g(J, "w", "same_in_both_readings"))
    # the shift
    for r in READINGS:
        for s in SCORE_COL:
            for k in FILES:
                for t in TASKS:
                    x, y = O["shift"][r][s][k][t], g(J, "shift", "ladder", r, s, k, t) or {}
                    for a, b in (("shift", "shift"), ("long", "part_long"), ("short", "part_short")):
                        T.num(f"{pth}: shift {r} {s} {k} {t} {b}", x[a], y.get(b))
                    T.num(f"{pth}: long share {r} {k} {t}", O["long_share"][r][k][t], y.get("long_share"))
    for s in SCORE_COL:
        for t in TASKS:
            for q in ("shift", "part_long", "part_short"):
                lo, hi = O["shift_panel"][s][t][q]
                T.num(f"{pth}: panel {q} {s} {t} min", lo, g(J, "shift", "panel", s, t, q, "min"))
                T.num(f"{pth}: panel {q} {s} {t} max", hi, g(J, "shift", "panel", s, t, q, "max"))
    # Qwen3.5-122B-A10B's requested replies
    if O["qwen"]:
        for t, x in O["qwen"].items():
            y = g(J, "qwen", "tasks", t) or {}
            for a, b in (("rows", "rows"), ("unreadable", "unreadable"), ("bare", "bare"),
                         ("bare_correct", "bare_content_correct"), ("unread_correct", "unreadable_content_correct"),
                         ("median", "bare_tokens_median")):
                T.num(f"{pth}: Qwen requested {t} {b}", x[a], y.get(b))
            for f_ in ("bare number", "bare option", "bare words", "box"):
                T.num(f"{pth}: Qwen requested {t} form {f_}", x["forms"].get(f_, 0), (y.get("forms") or {}).get(f_))
    else:
        T.n += 1      # without raw replies the forms are not read here; the line says so
    # the macros
    tex = root / NUMBERS_TEX
    have = {}
    if tex.exists():
        for mm in re.finditer(r"\\newcommand\{\\((?:lad|comb)[A-Za-z]*)\}\{(.*)\}", tex.read_text(encoding="utf-8")):
            have[mm.group(1)] = mm.group(2)
    want = ord_expected_macros(O, J)
    if not O["qwen"]:
        have = {k: v for k, v in have.items() if not k.startswith("ladQwenReq")}
    # names other macro modules write with the same prefixes (paper/macros_closed_ladder.py,
    # paper/macros_statement_values.py), read by name only; their values are those modules' (and their tests'), except the
    # ladder's sizes, compared below with this checker's own reading of the records
    other = other_lad_names(root)
    have_o = {k: v for k, v in have.items() if k in other and k not in want}
    have = {k: v for k, v in have.items() if k not in have_o}
    for k in sorted(set(want) | set(have)):
        T.text(f"{NUMBERS_TEX}: \\{k}", want.get(k, "(not expected)"), have.get(k, "(absent)"))
    for k, v in size_macros(root).items():
        if k in have_o:          # printed by paper/macros_closed_ladder.py; compared where numbers.tex holds it
            T.text(f"{NUMBERS_TEX}: \\{k}", v, have_o.get(k, "(absent)"))
    return T.n - n0


OTHER_LAD = re.compile(r"(lad|comb)[A-Za-z]*$")        # the names other_lad_names collects
MACRO_LIKE = re.compile(r"(lad|comb)[A-Z][A-Za-z]*$")  # a string in a module's source that would be such a macro


class SourceNotRead(ValueError):
    """A macro module's source has a form other_lad_names does not read; the checker stops on it
    because it cannot tell which names the module writes."""


def _tree(p: Path):
    """A module's syntax tree: its source parsed as text, nothing of it run; None where the file is absent."""
    return ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) if p.exists() else None


def _str(n):
    return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None


def _lad_key(t):
    """The name in an assignment target out["lad..."], or None."""
    return (_str(t.slice) if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and _str(t.slice)
            and OTHER_LAD.match(_str(t.slice)) else None)


def _written(node) -> list:
    """Every out["lad..."] = ... under node, at any depth."""
    return [k for n in ast.walk(node) if isinstance(n, ast.Assign) for t in n.targets for k in [_lad_key(t)] if k]


def _no_hidden_names(tree, read: list, where: str, skip=()) -> None:
    """Every macro-like lad/comb string in the source (outside the nodes in skip) is one of the out[...] assignments
    read, and no such name is built (f"lad{x}", "lad" + x, "lad%s" % x); otherwise the module may write names this
    reader does not see, and the checker stops."""
    skipped = {id(n) for s in skip for n in ast.walk(s)}
    seen = [_str(n) for n in ast.walk(tree) if id(n) not in skipped and _str(n) and MACRO_LIKE.match(_str(n))]
    part = re.compile(r"(lad|comb)[A-Za-z]*")
    built = [n.lineno for n in ast.walk(tree)
             if (isinstance(n, ast.JoinedStr) and n.values and _str(n.values[0]) is not None
                 and part.fullmatch(_str(n.values[0])))
             or (isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mod)) and _str(n.left) is not None
                 and part.fullmatch(_str(n.left).split("%")[0]))]
    if sorted(seen) != sorted(read) or built:
        raise SourceNotRead(f"{where}: lad/comb strings {sorted(set(seen) - set(read))} not read as out[...] "
                            f"assignments; names built at lines {built}")


def _load_path(root: Path, call):
    """_load(root, "a", "b", ...) -> root/a/b; None for any other expression."""
    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_load" and call.args
            and all(_str(a) is not None for a in call.args[1:])):
        return root.joinpath(*[_str(a) for a in call.args[1:]])
    return None


def _closed_ladder_names(root: Path) -> set:
    """The lad names paper/macros_closed_ladder.py writes, read from its source: closed_ladder_results(root) reads each file
    with _load (None where absent) and returns None while one of its `if x is None: return None` files is absent;
    lad_macros(R) writes out["lad..."] at its top level, stops at `if v is None: return out` where v's file is absent, and
    writes the names under `if v is not None [and w is not None ...]:` only where all those files exist. A name counts
    where it would be written from the files on disk, as when the module runs; any other form stops the checker."""
    p = root / "paper" / "macros_closed_ladder.py"
    tree = _tree(p)
    if tree is None:
        return set()
    fns = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    if "closed_ladder_results" not in fns or "lad_macros" not in fns:
        raise SourceNotRead(f"{p}: closed_ladder_results or lad_macros not found")
    _no_hidden_names(tree, _written(fns["lad_macros"]), str(p))
    # closed_ladder_results: R's keys -> the files they hold
    var, key_file = {}, {}
    for st in fns["closed_ladder_results"].body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            if _load_path(root, st.value) is None:
                raise SourceNotRead(f"{p}, line {st.lineno}: closed_ladder_results assigns other than _load(root, ...)")
            var[st.targets[0].id] = _load_path(root, st.value)
        elif isinstance(st, ast.If):
            t = st.test
            if not (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id in var
                    and isinstance(t.ops[0], ast.Is) and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value is None and len(st.body) == 1 and isinstance(st.body[0], ast.Return)
                    and not st.orelse):
                raise SourceNotRead(f"{p}, line {st.lineno}: closed_ladder_results has a test other than `if x is None`")
            if not var[t.left.id].exists():
                return set()
        elif isinstance(st, ast.Return) and isinstance(st.value, ast.Dict):
            for k, v in zip(st.value.keys, st.value.values):
                key_file[_str(k)] = (var.get(v.id) if isinstance(v, ast.Name) else _load_path(root, v))
        elif not (isinstance(st, ast.Expr) and _str(st.value) is not None):   # a docstring
            raise SourceNotRead(f"{p}, line {st.lineno}: closed_ladder_results has a statement not read here")

    def r_key(n):   # R["key"] -> "key"
        return (_str(n.slice) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "R"
                else None)

    def present(v):
        f = key_file.get(env.get(v))
        if f is None:
            raise SourceNotRead(f"{p}: lad_macros tests {v}, which holds no file of closed_ladder_results")
        return f.exists()

    def not_none_vars(t):   # `v is not None [and w is not None ...]` -> [v, w, ...]; None for any other test
        parts = t.values if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) else [t]
        out = []
        for c in parts:
            if not (isinstance(c, ast.Compare) and isinstance(c.left, ast.Name) and len(c.ops) == 1
                    and isinstance(c.ops[0], ast.IsNot) and isinstance(c.comparators[0], ast.Constant)
                    and c.comparators[0].value is None):
                return None
            out.append(c.left.id)
        return out

    env, names = {}, set()
    for st in fns["lad_macros"].body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if _lad_key(t):
                    names.add(_lad_key(t))
                elif isinstance(t, ast.Name) and r_key(st.value):
                    env[t.id] = r_key(st.value)
                elif isinstance(t, ast.Tuple) and isinstance(st.value, ast.Tuple):
                    env.update({a.id: r_key(b) for a, b in zip(t.elts, st.value.elts)
                                if isinstance(a, ast.Name) and r_key(b)})
            continue
        if isinstance(st, ast.If) and not _written(st):
            t = st.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id in env and len(t.ops) == 1
                    and isinstance(t.ops[0], ast.Is) and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value is None and isinstance(st.body[-1], ast.Return)):
                if not present(t.left.id):
                    break           # `if v is None: return out`: nothing after it is written
            continue                # a check that raises; it writes no name
        if isinstance(st, ast.If):
            vs = not_none_vars(st.test)
            if vs is None or st.orelse:
                raise SourceNotRead(f"{p}, line {st.lineno}: lad names under a test other than `v is not None`")
            if all(present(v) for v in vs):
                names |= set(_written(st))
            continue
        if _written(st):
            raise SourceNotRead(f"{p}, line {st.lineno}: lad names written inside a {type(st).__name__}")
    return names


def _statement_values_names(root: Path) -> set:
    """The lad and comb names paper/macros_statement_values.py writes: the keys of block "macros" of the JSON its PATH
    (os.path.join of strings, read from its source) names, as statement_values_results and statement_values return them."""
    p = root / "paper" / "macros_statement_values.py"
    tree = _tree(p)
    if tree is None:
        return set()
    paths = [st.value for st in tree.body if isinstance(st, ast.Assign) and len(st.targets) == 1
             and isinstance(st.targets[0], ast.Name) and st.targets[0].id == "PATH"]
    v = paths[-1] if paths else None
    if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "join" and v.args
            and all(_str(a) is not None for a in v.args)):
        raise SourceNotRead(f"{p}: PATH is not os.path.join of strings")
    prefixes = [st for st in tree.body if isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name) and st.targets[0].id == "PREFIXES"]
    _no_hidden_names(tree, [], str(p), skip=prefixes)   # its names are the JSON's; PREFIXES only checks them
    j = root.joinpath(*[_str(a) for a in v.args])
    if not j.exists():
        return set()
    R = json.loads(j.read_text(encoding="utf-8"))
    return {k for k in (R["macros"] if R else {}) if OTHER_LAD.match(k)}


def other_lad_names(root: Path) -> set:
    """The lad... and comb... names the other macro modules write (paper/macros_closed_ladder.py,
    paper/macros_statement_values.py), names only; empty where a module or its file is absent. Their sources are read as
    text (parsed, nothing of them imported or run) and the files they read are opened here, so the checker runs none
    of the project's code."""
    return _closed_ladder_names(root) | _statement_values_names(root)


def size_macros(root: Path) -> dict:
    """The ladder's published-size macros and V4-Flash's, from this checker's own reading of the
    records (size_facts), printed as the paper's B values."""
    ds, ml = size_facts(root, DS), size_facts(root, "mistral-large-2512")
    if not ds.get("published") or not ml.get("published"):
        return {}

    def b(x):
        return None if x is None else f"{Fraction(x).numerator}" if Fraction(x).denominator == 1 else str(x)
    return {"ladSizeDeepSeekTotalB": b(ds.get("total")), "ladSizeDeepSeekActiveInB": b(ds.get("active_input")),
            "ladSizeDeepSeekActiveOutB": b(ds.get("active_output")),
            "ladSizeMistralLargeActiveB": b(ml.get("active")),
            "ladSizeDeepSeekB": b((ds.get("predecessor") or {}).get("total"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", default=None, help="ladder.json (default <root>/results/ladder/analysis/ladder.json)")
    ap.add_argument("--nebius-md", default=None,
                    help="DeepSeek's report (default <root>/results/frontier_panel/nebius_panel.md)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    lj = Path(a.json) if a.json else root / "results/ladder/analysis/ladder.json"
    md = Path(a.nebius_md) if a.nebius_md else root / "results/frontier_panel/nebius_panel.md"
    _, m = run(root, lj, md)
    return 0 if m == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
