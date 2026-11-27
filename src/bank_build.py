r"""Machinery that turns prompt-bank design choices into a candidate bank.

This module is pure Python (standard library only) so that the checks run in the project environment without
PromptSuite. scripts/build_bank_candidates.py runs inside `.venv-promptsuite`, calls PromptSuite to generate
skeleton variants, and hands the variants to the functions here. Nothing in this module chooses a wording, a
variation type, a seed or a sampling law: those values come from config/promptsuite.yaml, which is
PENDING_REVIEW, and every function refuses missing values.

The paired design (direct versus zero-shot
chain-of-thought) uses one skeleton per pair. A skeleton is one pair (instruction variant, prompt-format variant). It carries three
placeholders, each exactly once across the pair:

    {question}       the only item variable; stays in the bank and is filled by scripts/build_requests.py
    {arm_clause}     filled here with the direct_answer or zero_shot_cot text
    {answer_format}  filled here with the requested final-answer format, identical in both arms

The two arms of a configuration therefore differ only in the arm clause. PromptSuite puts the instruction in a
system message and the item text in a user message; the messages built here mirror its `conversation` field
(system message only when the instruction is non-empty, user message stripped), and the script compares them
with PromptSuite's own conversation for rows that carry the real clause texts.

Output is a candidate bank in the src/d2_io.py schema. Freezing a candidate into manifests/gsm8k_bank.json is
a separate step, and nothing here writes there.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROMPTSUITE_VERSION = "3.0.7"            # the supported version; other versions are refused
GENERATOR = "PromptSuite"
EXAMPLE_LABEL = "EXAMPLE, not a registered bank"

STATUS_PENDING, STATUS_CONFIRMED = "PENDING_REVIEW", "confirmed"

ARMS = ("direct_answer", "zero_shot_cot")        # src/d2_io.py ARMS
PH_QUESTION, PH_ARM, PH_FORMAT = "{question}", "{arm_clause}", "{answer_format}"
PLACEHOLDERS = (PH_QUESTION, PH_ARM, PH_FORMAT)
ARM_FILL_PLACEHOLDERS = (PH_ARM, PH_FORMAT)
_FILL_RE = re.compile(r"\{(arm_clause|answer_format)\}")
_STRAY_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

# Variation types the builder accepts for the two template fields.
VT_FORMAT, VT_NOISE, VT_PARAPHRASE = "format structure", "typos and noise", "paraphrase_with_llm"
VARIATION_TYPES = (VT_FORMAT, VT_NOISE, VT_PARAPHRASE)
LLM_VARIATION_TYPES = (VT_PARAPHRASE,)
# PromptSuite 3.0.7 accepts these in a template but the builder refuses them (reason in the message).
REFUSED_VARIATION_TYPES = {
    "context": "the PromptSuite factory ignores the configured platform, model and key for it, calls the TogetherAI "
               "default model with the environment key, catches every API error and returns the unmodified prompt, "
               "so the result cannot be pinned or audited",
    "multidoc": "it has no augmenter in PromptSuite 3.0.7 and silently falls through to typos and noise",
    "shuffle": "it acts on list-valued row columns and needs a gold index; GSM8K has none",
    "enumerate": "it acts on list-valued row columns; GSM8K has none",
}

SAMPLING_LAWS = ("uniform",)

# Environment variables PromptSuite 3.0.7 reads API keys from (promptsuite.shared.constants).
PLATFORM_KEY_VARS = {"TogetherAI": "TOGETHER_API_KEY", "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY",
                     "Google": "GOOGLE_API_KEY", "Cohere": "COHERE_API_KEY"}
GENERIC_KEY_VAR = "API_KEY"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RESERVED_NAMES = ("gsm8k_bank",)
LOG_PROBLEM_RE = re.compile(r"⚠|\bError\b|\berror\b|Traceback|\bWarning\b|fallback", re.UNICODE)

BLOCK_KEYS = ("name", "seed", "variations_per_field", "max_variations_per_row", "instruction", "prompt_format",
              "arm_clauses", "answer_format", "sampling_law", "llm", "required_literals")
TOP_KEYS = ("status", "example") + BLOCK_KEYS
NESTED_KEYS = {"instruction": ("text", "variation_types"), "prompt_format": ("text", "variation_types"),
               "arm_clauses": ARMS, "answer_format": ("scorer_candidate", "text"),
               "llm": ("allowed", "platform", "model")}
# Keys a nested block may also hold, with the value an absent key stands for.
OPTIONAL_NESTED_KEYS = {"instruction": {"placement": "system"}, "llm": {"record": None, "review": None}}
# Where the instruction goes: PromptSuite's `conversation` (a system message) or PromptSuite's `prompt` string (the
# instruction and the prompt format joined by a blank line, sent as the one user message).
PLACEMENTS = ("system", "user")
# llm.platform of a paraphrase answered by a pinned open model on a GPU machine (config/paraphraser.yaml) and replayed
# from its saved record (scripts/paraphrase_d2.py). No hosted API is called and no key is read.
REPLAY_PLATFORM = "local_replay"
RECORD_KIND, REVIEW_KIND = "paraphrase_record", "paraphrase_review"
VERDICTS = ("keep", "drop")


class BankBuildError(Exception):
    """An input does not qualify; the script prints it as REFUSED and exits 2. Nothing is written."""


# --------------------------------------------------------------------------------------------
# Placeholder checks
# --------------------------------------------------------------------------------------------

def count_placeholders(texts: Iterable[str | None]) -> dict[str, int]:
    """Occurrences of each of the three placeholders across the given texts (None counts as empty)."""
    joined = "\n".join(t for t in texts if t)
    return {ph: joined.count(ph) for ph in PLACEHOLDERS}


def skeleton_problems(instruction: str | None, prompt_format: str | None) -> list[str]:
    """Problems of one skeleton: each placeholder must occur exactly once across instruction and prompt format."""
    counts = count_placeholders([instruction, prompt_format])
    return [f"{ph} occurs {n} times across the instruction and the prompt format (exactly once is required)"
            for ph, n in counts.items() if n != 1]


def stray_placeholders(texts: Iterable[str | None]) -> list[str]:
    """Brace tokens of the form {word} that are not one of the three placeholders (reported, not dropped)."""
    found: list[str] = []
    for t in texts:
        found += [m for m in _STRAY_RE.findall(t or "") if m not in PLACEHOLDERS]
    return sorted(set(found))


def missing_literals(texts: Iterable[str | None], required: Sequence[str] | None) -> list[str]:
    """Required literals that occur in none of the texts."""
    if not required:
        return []
    joined = "\n".join(t for t in texts if t)
    return [lit for lit in required if lit not in joined]


def fill_arm(text: str, arm_clause: str, answer_format: str) -> str:
    """Fill {arm_clause} and {answer_format} in one pass, so that inserted text is never scanned again.
    {question} is left alone."""
    values = {"arm_clause": arm_clause, "answer_format": answer_format}
    return _FILL_RE.sub(lambda m: values[m.group(1)], text)


def build_messages(instruction: str | None, prompt_format: str, arm_clause: str, answer_format: str,
                   placement: str = "system") -> list[dict]:
    """Messages for one arm of one skeleton. With placement system they are PromptSuite's `conversation`: a system
    message when the instruction is non-empty (not stripped), then the user message (stripped). With placement user
    the one user message is PromptSuite's `prompt` string: the instruction and the user text joined by a blank line."""
    user = fill_arm(prompt_format, arm_clause, answer_format).strip()
    if instruction and placement == "user":
        return [{"role": "user", "content": fill_arm(instruction, arm_clause, answer_format) + "\n\n" + user}]
    msgs: list[dict] = []
    if instruction:
        msgs.append({"role": "system", "content": fill_arm(instruction, arm_clause, answer_format)})
    msgs.append({"role": "user", "content": user})
    return msgs


def placement_of(block: Mapping) -> Any:
    return (block.get("instruction") or {}).get("placement", OPTIONAL_NESTED_KEYS["instruction"]["placement"])


def filled_problems(messages: Sequence[Mapping]) -> list[str]:
    """After the arm fill {question} occurs exactly once, no other placeholder is left, and the last role is user."""
    text = "\n".join(m["content"] for m in messages)
    out: list[str] = []
    n = text.count(PH_QUESTION)
    if n != 1:
        out.append(f"{PH_QUESTION} occurs {n} times after the arm fill (exactly once is required)")
    for ph in ARM_FILL_PLACEHOLDERS:
        if ph in text:
            out.append(f"{ph} is left unfilled after the arm fill")
    if not messages or messages[-1]["role"] != "user":
        out.append("the last message must have role user")
    return out


# --------------------------------------------------------------------------------------------
# Answer format and scorer candidate
# --------------------------------------------------------------------------------------------

def format_prefix_literal(template: str) -> str:
    """The literal text a scorer template requires before the number, with the regex escapes removed.
    'Answer:[ \\t]<NUM>' gives 'Answer:'; '\\\\boxed\\{<NUM>\\}' gives '\\boxed{'."""
    head = template.split("<NUM>", 1)[0]
    head = head.replace("[ \\t]+", " ").replace("[ \\t]", " ")
    return re.sub(r"\\(.)", r"\1", head).strip()


def answer_format_problem(text: str, scorer_template: str, case_sensitive: bool) -> str | None:
    """None when the requested-format text contains the literal the scorer candidate extracts, else a message."""
    lit = format_prefix_literal(scorer_template)
    if not lit:
        return None
    hay, needle = (text, lit) if case_sensitive else (text.casefold(), lit.casefold())
    if needle in hay:
        return None
    return (f"the answer-format text does not contain {lit!r}, the literal the scorer candidate extracts "
            f"(template {scorer_template!r}, case {'sensitive' if case_sensitive else 'insensitive'})")


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

def check_status(cfg: Any, use_example: bool) -> None:
    """Refuse while the configuration is PENDING_REVIEW, unless the example block is requested."""
    if not isinstance(cfg, dict):
        raise BankBuildError("the configuration must be a YAML mapping")
    if use_example:
        return
    status = cfg.get("status")
    if status == STATUS_PENDING:
        raise BankBuildError("config/promptsuite.yaml has status PENDING_REVIEW: the bank design choices "
                             "are not made. Fill in the fields and set status to 'confirmed', or run with --example "
                             "to build from the labelled example values (the output is marked as not registered)")
    if status != STATUS_CONFIRMED:
        raise BankBuildError(f"config status must be {STATUS_PENDING!r} or {STATUS_CONFIRMED!r}, got {status!r}")


def active_block(cfg: Mapping, use_example: bool) -> dict:
    block = cfg.get("example") if use_example else cfg
    if not isinstance(block, dict):
        raise BankBuildError("the configuration has no 'example' block" if use_example else "the configuration is not a mapping")
    return block


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _types_problems(where: str, types: Any) -> list[str]:
    if not isinstance(types, list):
        return [f"{where}.variation_types must be a list (empty for no variation), got {types!r}"]
    out: list[str] = []
    for t in types:
        if isinstance(t, str) and t in REFUSED_VARIATION_TYPES:
            out.append(f"{where}.variation_types: {t!r} is refused: {REFUSED_VARIATION_TYPES[t]}")
        elif t not in VARIATION_TYPES:
            out.append(f"{where}.variation_types: {t!r} is not supported; supported: {list(VARIATION_TYPES)}")
    if len(set(map(str, types))) != len(types):
        out.append(f"{where}.variation_types lists a type twice")
    return out


def llm_types_used(block: Mapping) -> list[str]:
    used: list[str] = []
    for field in ("instruction", "prompt_format"):
        used += [t for t in (block.get(field) or {}).get("variation_types") or [] if t in LLM_VARIATION_TYPES]
    return sorted(set(used))


def validate_config(cfg: Any, use_example: bool,
                    candidates: Mapping[str, tuple[str, bool]]) -> list[str]:
    """Every problem of the active block, all at once. `candidates` maps a scorer candidate name to
    (template, case_sensitive). Missing values (None) are problems: a null is a choice not yet made."""
    if not isinstance(cfg, dict):
        return ["the configuration must be a YAML mapping"]
    out: list[str] = [f"unknown top-level key {k!r}" for k in sorted(set(cfg) - set(TOP_KEYS), key=str)]
    if use_example and not isinstance(cfg.get("example"), dict):
        return out + ["the configuration has no 'example' block"]
    block = cfg["example"] if use_example else cfg
    where = "example." if use_example else ""
    if use_example:
        out += [f"example: unknown key {k!r}" for k in sorted(set(block) - set(BLOCK_KEYS), key=str)]
    for k in BLOCK_KEYS:
        if k not in block:
            out.append(f"{where}{k} is missing")
    for k, sub in NESTED_KEYS.items():
        v = block.get(k)
        if k in block and not isinstance(v, dict):
            out.append(f"{where}{k} must be a mapping with {list(sub)}")
        elif isinstance(v, dict):
            allowed = set(sub) | set(OPTIONAL_NESTED_KEYS.get(k, {}))
            out += [f"{where}{k}: unknown key {x!r}" for x in sorted(set(v) - allowed, key=str)]
            out += [f"{where}{k}.{x} is missing" for x in sub if x not in v]
    if out:
        return out

    def g(*path):
        v = block
        for p in path:
            v = v[p]
        return v

    name = block["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        out.append(f"{where}name must match {NAME_RE.pattern} (it names the output file), got {name!r}")
    elif name in RESERVED_NAMES:
        out.append(f"{where}name {name!r} is reserved for the frozen bank")
    seed = block["seed"]
    if not _is_int(seed) or not 0 <= seed < 2**32:
        out.append(f"{where}seed must be an integer in [0, 2**32), got {seed!r}")
    vpf = block["variations_per_field"]
    if not _is_int(vpf) or vpf < 1:
        out.append(f"{where}variations_per_field must be an integer >= 1, got {vpf!r}")
    cap = block["max_variations_per_row"]
    if cap is not None and (not _is_int(cap) or cap < 1):
        out.append(f"{where}max_variations_per_row must be null (no cap) or an integer >= 1, got {cap!r}")

    itext, ptext = g("instruction", "text"), g("prompt_format", "text")
    if not isinstance(itext, str):
        out.append(f"{where}instruction.text must be a string (empty for no system message), got {itext!r}")
        itext = None
    if not _nonempty_str(ptext):
        out.append(f"{where}prompt_format.text must be a non-empty string, got {ptext!r}")
        ptext = None
    for field in ("instruction", "prompt_format"):
        out += _types_problems(where + field, g(field, "variation_types"))
    if itext == "" and g("instruction", "variation_types"):
        out.append(f"{where}instruction.variation_types is set but instruction.text is empty")
    placement = placement_of(block)
    if placement not in PLACEMENTS:
        out.append(f"{where}instruction.placement must be one of {list(PLACEMENTS)}, got {placement!r}")
    elif placement == "user" and itext == "":
        out.append(f"{where}instruction.placement is user but instruction.text is empty")
    if itext is not None and ptext is not None:
        out += [f"{where}template: {p}" for p in skeleton_problems(itext, ptext)]

    clauses = {a: g("arm_clauses", a) for a in ARMS}
    for a, c in clauses.items():
        if not _nonempty_str(c):
            out.append(f"{where}arm_clauses.{a} must be a non-empty string, got {c!r}")
    if all(_nonempty_str(c) for c in clauses.values()) and clauses[ARMS[0]] == clauses[ARMS[1]]:
        out.append(f"{where}arm_clauses: the two arms have the same text, so there is no contrast")

    cand = g("answer_format", "scorer_candidate")
    ftext = g("answer_format", "text")
    if not isinstance(cand, str) or cand not in candidates:
        out.append(f"{where}answer_format.scorer_candidate {cand!r} is not a name in src/scoring.py CANDIDATE_SPECS: "
                   f"{sorted(candidates)}")
    if not _nonempty_str(ftext):
        out.append(f"{where}answer_format.text must be a non-empty string, got {ftext!r}")
    elif isinstance(cand, str) and cand in candidates:
        msg = answer_format_problem(ftext, *candidates[cand])
        if msg:
            out.append(f"{where}answer_format: {msg}")

    for label, text in (("arm_clauses.direct_answer", clauses[ARMS[0]]), ("arm_clauses.zero_shot_cot", clauses[ARMS[1]]),
                        ("answer_format.text", ftext)):
        if isinstance(text, str):
            bad = [ph for ph in PLACEHOLDERS if ph in text]
            if bad:
                out.append(f"{where}{label} contains the placeholder(s) {bad}; the texts that fill placeholders may not")

    law = block["sampling_law"]
    if law not in SAMPLING_LAWS:
        out.append(f"{where}sampling_law must be one of {list(SAMPLING_LAWS)} (only these are implemented), got {law!r}")

    llm = block["llm"]
    if not isinstance(llm["allowed"], bool):
        out.append(f"{where}llm.allowed must be true or false, got {llm['allowed']!r}")
    else:
        used = llm_types_used(block)
        if used and not llm["allowed"]:
            out.append(f"{where}variation types {used} call an LLM but llm.allowed is false")
        replay = llm["platform"] == REPLAY_PLATFORM
        if llm["allowed"] and used:
            if not replay and (not isinstance(llm["platform"], str) or llm["platform"] not in PLATFORM_KEY_VARS):
                out.append(f"{where}llm.platform must be {REPLAY_PLATFORM!r} or one of {sorted(PLATFORM_KEY_VARS)}, "
                           f"got {llm['platform']!r}")
            if not _nonempty_str(llm["model"]):
                out.append(f"{where}llm.model must be set when an LLM variation type is used")
            if replay:
                out += [f"{where}llm.{x} must name a file when llm.platform is {REPLAY_PLATFORM!r}"
                        for x in ("record", "review") if not _nonempty_str(llm.get(x))]
        for x in ("record", "review"):
            if llm.get(x) is not None and not (replay and used and llm["allowed"]):
                out.append(f"{where}llm.{x} applies only to llm.platform {REPLAY_PLATFORM!r} with an allowed LLM "
                           "variation type")

    req = block["required_literals"]
    if req is not None and not (isinstance(req, list) and all(_nonempty_str(x) for x in req)):
        out.append(f"{where}required_literals must be null or a list of non-empty strings, got {req!r}")
    return out


def make_template(block: Mapping) -> dict:
    """The template dict handed to PromptSuite (instruction as the system message, prompt format as the user message)."""
    t: dict = {}
    if block["instruction"]["text"]:
        t["instruction"] = block["instruction"]["text"]
        if block["instruction"]["variation_types"]:
            t["instruction variations"] = list(block["instruction"]["variation_types"])
    t["prompt format"] = block["prompt_format"]["text"]
    if block["prompt_format"]["variation_types"]:
        t["prompt format variations"] = list(block["prompt_format"]["variation_types"])
    return t


# --------------------------------------------------------------------------------------------
# LLM gate and environment
# --------------------------------------------------------------------------------------------

def llm_gate(block: Mapping, allow_flag: bool, environ: Mapping[str, str]) -> list[str]:
    """Reasons the builder must refuse before importing PromptSuite. Empty when no LLM variation type is used,
    or when the configuration allows it, --allow-llm-api is given, and the platform's key is in the environment."""
    used = llm_types_used(block)
    if not used:
        return []
    out: list[str] = []
    llm = block["llm"]
    if not llm["allowed"]:
        out.append("an LLM variation type is configured but llm.allowed is false")
    if llm.get("platform") == REPLAY_PLATFORM:
        return out                     # replayed from a saved record: no API, no key (replay_problems checks it)
    if not allow_flag:
        out.append(f"variation types {used} call a hosted LLM; pass --allow-llm-api to permit it (it is a paid, "
                   "unseeded call)")
    platform = llm.get("platform")
    var = PLATFORM_KEY_VARS.get(platform) if isinstance(platform, str) else None
    if var is None:
        out.append(f"llm.platform {llm.get('platform')!r} is not one of {sorted(PLATFORM_KEY_VARS)}")
    elif not environ.get(var):
        out.append(f"no API key in the environment variable {var}; keys stay in the environment, never in files "
                   "(.env files are not read)")
    return out


def scrub_api_env(environ: dict) -> list[str]:
    """Remove every API-key variable PromptSuite reads from `environ`; returns the names that were present."""
    names = list(PLATFORM_KEY_VARS.values()) + [GENERIC_KEY_VAR]
    return [n for n in names if environ.pop(n, None) is not None]


def log_problems(log_text: str) -> list[str]:
    """Lines of PromptSuite's captured output that report a warning, a fallback or an error."""
    return [ln.strip() for ln in log_text.splitlines() if LOG_PROBLEM_RE.search(ln)]


# --------------------------------------------------------------------------------------------
# Variants, skeletons, bank
# --------------------------------------------------------------------------------------------

def canonical_variants(variants: Iterable[str], original: str | None) -> list[str]:
    """Unique variants in a hash-independent order: the original text first (when present), then code-point order."""
    uniq = set(variants)
    head = [original] if original is not None and original in uniq else []
    return head + sorted(uniq - set(head))


def variant_sha256(instruction_variants: Iterable[str], prompt_format_variants: Iterable[str]) -> str:
    """sha256 over the sorted unique variant strings of both fields."""
    body = {"instruction": sorted(set(instruction_variants)), "prompt_format": sorted(set(prompt_format_variants))}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def config_id(i: int, j: int, width: int) -> str:
    return f"ps-i{i:0{width}d}-f{j:0{width}d}"


def build_skeletons(pairs: Iterable[tuple[str, str]], block: Mapping,
                    rejected: Mapping[str, str] | None = None) -> dict:
    """Deduplicate, check and fill the skeleton pairs read from PromptSuite's `field_values`.

    `pairs` holds (instruction variant, prompt-format variant) in any order. Returns the canonical variant lists, the
    kept skeletons with both arms' messages (sorted by variant indices), and the dropped skeletons with reasons.
    A skeleton is dropped when a placeholder is lost or duplicated, when a required literal is missing, when the
    filled messages are not valid, when its filled messages repeat those of an earlier skeleton, or when its
    instruction variant is in `rejected` (variant text to the reviewer's note, from review_rejections)."""
    itext = block["instruction"]["text"] or None
    ptext = block["prompt_format"]["text"]
    uniq = list(dict.fromkeys((i if i is not None else "", p) for i, p in pairs))
    if not uniq:
        raise BankBuildError("PromptSuite returned no skeleton variants")
    iv = canonical_variants([i for i, _ in uniq], itext)
    pv = canonical_variants([p for _, p in uniq], ptext)
    iidx, pidx = {t: n for n, t in enumerate(iv)}, {t: n for n, t in enumerate(pv)}
    clauses = block["arm_clauses"]
    fmt = block["answer_format"]["text"]
    req = block.get("required_literals")
    placement = placement_of(block)
    rejected = rejected or {}
    kept, dropped, seen = [], [], {}
    for (i, p) in sorted(uniq, key=lambda ip: (iidx[ip[0]], pidx[ip[1]])):
        ii, jj = iidx[i], pidx[p]
        reasons = skeleton_problems(i or None, p)
        lits = missing_literals([i, p], req)
        if lits:
            reasons.append(f"required literal(s) {lits} are missing")
        if i in rejected:
            reasons.append(f"instruction variant {ii} was dropped in the meaning review: {rejected[i]}")
        msgs = None
        if not reasons:
            msgs = {a: build_messages(i or None, p, clauses[a], fmt, placement) for a in ARMS}
            for a in ARMS:
                reasons += [f"arm {a}: {r}" for r in filled_problems(msgs[a])]
        if not reasons:
            key = json.dumps([msgs[a] for a in ARMS], ensure_ascii=False, sort_keys=True)
            if key in seen:
                reasons.append(f"filled messages are identical to those of instruction variant {seen[key][0]}, "
                               f"prompt-format variant {seen[key][1]}")
            else:
                seen[key] = (ii, jj)
        if reasons:
            dropped.append({"instruction_variant": ii, "prompt_format_variant": jj, "reasons": reasons})
        else:
            kept.append({"instruction_variant": ii, "prompt_format_variant": jj, "instruction": i, "prompt_format": p,
                         "messages": msgs})
    return {"instruction_variants": iv, "prompt_format_variants": pv, "kept": kept, "dropped": dropped,
            "n_pairs": len(uniq)}


def uniform_probabilities(ids: Sequence[str]) -> dict[str, float]:
    n = len(ids)
    if n < 1:
        raise BankBuildError("a bank needs at least one configuration")
    p = 1.0 / n
    probs = {i: p for i in ids}
    if abs(math.fsum(probs.values()) - 1.0) > 1e-9:
        raise BankBuildError("uniform probabilities do not sum to 1")
    return probs


def assemble_bank(block: Mapping, table: Mapping, *, generator_version: str, example: bool,
                  generation_record: Mapping) -> dict:
    """The candidate bank in the src/d2_io.py schema (generator, generator_version, seed, sampling_probabilities,
    configurations), plus `generation_record`, which d2_io carries along and ignores."""
    kept = table["kept"]
    if not kept:
        raise BankBuildError("no skeleton survived the checks; see the dropped list: "
                             + json.dumps(table["dropped"], ensure_ascii=False)[:2000])
    width = max(2, len(str(max(len(table["instruction_variants"]), len(table["prompt_format_variants"])) - 1)))
    itypes = list(block["instruction"]["variation_types"])
    ptypes = list(block["prompt_format"]["variation_types"])
    itext = block["instruction"]["text"] or None
    ptext = block["prompt_format"]["text"]
    configs: dict[str, dict] = {}
    for k in kept:
        cid = config_id(k["instruction_variant"], k["prompt_format_variant"], width)
        configs[cid] = {
            "dimensions": {
                "instruction_variant": k["instruction_variant"],
                "instruction_is_original": k["instruction"] == (itext or ""),
                "instruction_variation_types": itypes,
                "prompt_format_variant": k["prompt_format_variant"],
                "prompt_format_is_original": k["prompt_format"] == ptext,
                "prompt_format_variation_types": ptypes,
                "answer_format_scorer_candidate": block["answer_format"]["scorer_candidate"],
            },
            "arms": {a: {"messages": k["messages"][a]} for a in ARMS},
        }
    probs = uniform_probabilities(list(configs))
    generator = f"{GENERATOR} ({EXAMPLE_LABEL})" if example else GENERATOR
    return {"generator": generator, "generator_version": generator_version, "seed": block["seed"],
            "sampling_probabilities": probs, "configurations": configs, "generation_record": dict(generation_record)}


def canonical_bank_bytes(bank: Mapping) -> bytes:
    """Deterministic UTF-8 serialization (fixed key order as built, LF newlines, trailing newline)."""
    return (json.dumps(bank, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def output_path_problem(path: str | Path, repo_root: str | Path) -> str | None:
    """A candidate is never written under manifests/ (freezing is a separate step)."""
    p, root = Path(path).resolve(), Path(repo_root).resolve()
    manifests = root / "manifests"
    if p == manifests or manifests in p.parents:
        return f"{p} lies under manifests/; candidates go to data/bank_candidates/ and freezing is a separate step"
    if p.name == "gsm8k_bank.json":
        return "gsm8k_bank.json is the frozen bank's file name; a candidate must not use it"
    return None


# --------------------------------------------------------------------------------------------
# Replayed paraphrase: the saved record and the meaning review
# --------------------------------------------------------------------------------------------

def replay_used(block: Mapping) -> bool:
    return bool(llm_types_used(block)) and (block.get("llm") or {}).get("platform") == REPLAY_PLATFORM


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FENCE_RE = re.compile(r"\A```[A-Za-z0-9_+-]*[ \t]*\n(.*)\n```\Z", re.DOTALL)


def strip_code_fence(text: str) -> tuple[str, bool]:
    """The text inside one Markdown code fence that wraps the whole answer (surrounding whitespace ignored), and
    whether a fence was removed. Llama-3.3-70B wraps its Python list in such a fence, which PromptSuite 3.0.7's
    ast.literal_eval cannot parse; its hosted path would then fall back to the original text alone."""
    m = _FENCE_RE.match(text.strip())
    return (m.group(1), True) if m else (text, False)


def record_problems(record: Any, block: Mapping) -> list[str]:
    """Problems of a frozen paraphrase record (scripts/paraphrase_d2.py freeze) for this configuration."""
    if not isinstance(record, dict) or record.get("kind") != RECORD_KIND:
        return [f"the paraphrase record must be a mapping with kind {RECORD_KIND!r}"]
    out: list[str] = []
    req, resp, model = record.get("request") or {}, record.get("response") or {}, record.get("model") or {}
    if not isinstance(req.get("prompt_sha256"), str) or len(req["prompt_sha256"]) != 64:
        out.append("record.request.prompt_sha256 must be a sha256")
    text = resp.get("text")
    if not isinstance(text, str):
        out.append("record.response.text must be a string")
    elif resp.get("sha256") != sha256_text(text):
        out.append("record.response.sha256 does not match the response text")
    if resp.get("finish_reason") != "stop":
        out.append(f"record.response.finish_reason is {resp.get('finish_reason')!r}; a truncated answer is not replayed")
    rep = record.get("replay") or {}
    if not isinstance(rep.get("text"), str) or rep.get("sha256") != sha256_text(rep["text"]):
        out.append("record.replay must hold the replayed text and its sha256")
    elif isinstance(text, str):
        inner, fenced = strip_code_fence(text)
        if rep["text"] not in ((inner,) if fenced else (text,)) or bool(rep.get("code_fence_removed")) != fenced:
            out.append("record.replay.text must be the response, with only a wrapping Markdown code fence removed")
    if model.get("key") != block["llm"]["model"]:
        out.append(f"the record was generated by {model.get('key')!r}, but llm.model is {block['llm']['model']!r}")
    return out


def review_rejections(review: Any, record_sha256: str, variants: Sequence[str],
                      original: str | None) -> tuple[dict[str, str], list[str]]:
    """The hand review of the paraphrases. Returns {variant text: note} for the variants marked
    drop, and the problems. Every instruction variant other than the original needs exactly one verdict, keep or drop,
    and a drop needs a note. A verdict for a text that is not a variant is a problem."""
    if not isinstance(review, dict) or review.get("kind") != REVIEW_KIND:
        return {}, [f"the review must be a mapping with kind {REVIEW_KIND!r}"]
    out: list[str] = []
    if review.get("record_sha256") != record_sha256:
        out.append("the review names another paraphrase record (record_sha256 differs)")
    who = review.get("reviewers")
    if not (isinstance(who, list) and who and all(_nonempty_str(x) for x in who)):
        out.append("review.reviewers must list who reviewed the paraphrases")
    verdicts = review.get("verdicts")
    if not isinstance(verdicts, list):
        return {}, out + ["review.verdicts must be a list"]
    need = [v for v in variants if v != original]
    seen: dict[str, dict] = {}
    for k, v in enumerate(verdicts):
        if not isinstance(v, dict) or not isinstance(v.get("text"), str):
            out.append(f"review.verdicts[{k}] must be a mapping with a text")
            continue
        if v["text"] in seen:
            out.append(f"review.verdicts[{k}] repeats a text")
        seen[v["text"]] = v
        if v.get("verdict") not in VERDICTS:
            out.append(f"review.verdicts[{k}].verdict must be one of {list(VERDICTS)}, got {v.get('verdict')!r}")
        elif v["verdict"] == "drop" and not _nonempty_str(v.get("note")):
            out.append(f"review.verdicts[{k}] drops a variant without a note")
        if v["text"] not in need:
            out.append(f"review.verdicts[{k}] judges a text that is not a paraphrased instruction variant: {v['text']!r}")
    out += [f"no verdict for instruction variant {v!r}" for v in need if v not in seen]
    rejected = {t: str(v.get("note")) for t, v in seen.items() if v.get("verdict") == "drop" and t in need}
    return rejected, out
