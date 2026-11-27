"""Tests for src/bank_build.py and scripts/build_bank_candidates.py (candidate prompt-bank builder).

The pure checks run in the project environment. The tests that run PromptSuite call the isolated environment
`.venv-promptsuite` through a subprocess and are skipped when it does not exist. Nothing here calls an API: the
refusal tests run in process with PromptSuite blocked from importing, and no API key is ever set.
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import bank_build as bb, d2_io, scoring

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_bank_candidates.py"
CONFIG = ROOT / "config" / "promptsuite.yaml"
VENV_PY = ROOT / ".venv-promptsuite" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
needs_venv = pytest.mark.skipif(not VENV_PY.exists(), reason=".venv-promptsuite is not set up")

CANDIDATES = {n: (s.template, s.case_sensitive) for n, s in scoring.CANDIDATE_SPECS.items()}


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def example_block():
    return copy.deepcopy(load_config()["example"])


def confirmed_cfg(**over):
    """A confirmed configuration built from the example block (test data, not a decision)."""
    cfg = example_block()
    cfg["status"] = "confirmed"
    for k, v in over.items():
        cfg[k] = v
    return cfg


def pending_cfg():
    """A configuration as first shipped: PENDING_REVIEW, every choice null, the example block kept."""
    cfg = {"status": "PENDING_REVIEW", "example": example_block()}
    for k in bb.BLOCK_KEYS:
        cfg[k] = {s: None for s in bb.NESTED_KEYS[k]} if k in bb.NESTED_KEYS else None
    cfg["llm"]["allowed"] = False
    return cfg


def write_cfg(path, cfg):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_script():
    spec = importlib.util.spec_from_file_location("build_bank_candidates_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in list(bb.PLATFORM_KEY_VARS.values()) + [bb.GENERIC_KEY_VAR, "PYTHONHASHSEED", "PYTHON_DOTENV_DISABLED"]}
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def run_builder(args, cwd, **env_extra):
    return subprocess.run([str(VENV_PY), str(SCRIPT), *map(str, args)], cwd=str(cwd), env=clean_env(**env_extra),
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


# ---------------------------------------------------------------- placeholder checks

def test_count_placeholders_and_skeleton_problems():
    ok_i = "Solve. {arm_clause} {answer_format}"
    ok_p = "Q: {question}\nA:"
    assert bb.count_placeholders([ok_i, ok_p]) == {"{question}": 1, "{arm_clause}": 1, "{answer_format}": 1}
    assert bb.count_placeholders([None, ok_p])["{arm_clause}"] == 0
    assert bb.skeleton_problems(ok_i, ok_p) == []
    assert bb.skeleton_problems(None, "{arm_clause} {answer_format} {question}") == []     # all in the user message
    lost = bb.skeleton_problems("Solve. {answer_format}", ok_p)
    assert len(lost) == 1 and "{arm_clause}" in lost[0] and "0 times" in lost[0]
    dup = bb.skeleton_problems(ok_i, ok_p + " {question}")
    assert len(dup) == 1 and "{question}" in dup[0] and "2 times" in dup[0]
    assert len(bb.skeleton_problems("Solve.", "Q:")) == 3                                  # everything lost
    mangled = bb.skeleton_problems("Solve. {arm_clause} {answer_format}", "Q: {qeustion}")
    assert len(mangled) == 1 and "{question}" in mangled[0]                                # a mangled placeholder is a lost one


def test_stray_placeholders_and_missing_literals():
    assert bb.stray_placeholders(["a {arm_clause} {x}", "b {question} {y_1} \\boxed{}", None]) == ["{x}", "{y_1}"]
    assert bb.stray_placeholders(["{question}", "{arm_clause}", "{answer_format}"]) == []
    assert bb.missing_literals(["Answer: x", "y"], ["Answer:", "N"]) == ["N"]
    assert bb.missing_literals(["Answer:"], None) == [] and bb.missing_literals(["x"], []) == []


def test_fill_arm_is_single_pass_and_leaves_question_and_other_braces():
    out = bb.fill_arm("A {arm_clause} B {answer_format} C {question} D {x} \\boxed{} {{y}}", "X {answer_format}", "F")
    assert out == "A X {answer_format} B F C {question} D {x} \\boxed{} {{y}}"
    assert bb.fill_arm("{arm_clause}{arm_clause}", "z", "f") == "zz"


def test_build_messages_mirror_promptsuite_conversation():
    msgs = bb.build_messages("Sys {arm_clause} {answer_format}", "  Q: {question}\nA:  \n", "CLAUSE", "FMT")
    assert msgs == [{"role": "system", "content": "Sys CLAUSE FMT"}, {"role": "user", "content": "Q: {question}\nA:"}]
    assert bb.build_messages(None, "{arm_clause} {answer_format} {question}", "c", "f") == [
        {"role": "user", "content": "c f {question}"}]
    assert bb.build_messages("", "{question}", "c", "f") == [{"role": "user", "content": "{question}"}]
    sys_text = " trailing space kept "
    assert bb.build_messages(sys_text, "{question}", "c", "f")[0]["content"] == sys_text      # the system text is not stripped


def test_filled_problems():
    good = [{"role": "system", "content": "s"}, {"role": "user", "content": "q {question}"}]
    assert bb.filled_problems(good) == []
    assert any("2 times" in p for p in bb.filled_problems([{"role": "user", "content": "{question} {question}"}]))
    assert any("0 times" in p for p in bb.filled_problems([{"role": "user", "content": "none"}]))
    assert any("{arm_clause}" in p for p in bb.filled_problems([{"role": "user", "content": "{question} {arm_clause}"}]))
    assert any("last message" in p for p in bb.filled_problems([{"role": "user", "content": "{question}"},
                                                                 {"role": "assistant", "content": "a"}]))


# ---------------------------------------------------------------- answer format and scorer candidates

def test_format_prefix_literal_matches_every_scorer_candidate():
    accepted = {"the_answer_is": "So the answer is 18", "the_answer_is_markup": "So the answer is 18",
                "hash_n": "#### 18", "answer_colon": "Answer: 18", "answer_colon_markup": "Answer: 18",
                "boxed": "\\boxed{18}"}
    assert set(accepted) == set(scoring.CANDIDATE_SPECS), "a new scorer candidate needs a line here"
    for name, spec in scoring.CANDIDATE_SPECS.items():
        lit = bb.format_prefix_literal(spec.template)
        assert lit and lit in accepted[name], (name, lit)
        assert scoring.strict_extract(accepted[name], spec) == "18"           # the sample really is accepted by the scorer
        assert bb.answer_format_problem(f"Finish with {lit} <number>.", spec.template, spec.case_sensitive) is None
        assert bb.answer_format_problem("Give the number at the end.", spec.template, spec.case_sensitive)
    assert bb.format_prefix_literal(scoring.CANDIDATE_SPECS["boxed"].template) == "\\boxed{"
    # case handling: the_answer_is is case-insensitive, answer_colon is not
    assert bb.answer_format_problem("THE ANSWER IS 18", *CANDIDATES["the_answer_is"]) is None
    assert bb.answer_format_problem("answer: 18", *CANDIDATES["answer_colon"])


# ---------------------------------------------------------------- configuration

def test_shipped_config_is_the_choice_of_30_october_with_the_registered_format():
    cfg = load_config()
    assert cfg["status"] == "confirmed" and set(cfg) == set(bb.TOP_KEYS)
    bb.check_status(cfg, use_example=False)
    assert cfg["instruction"] == {"text": "Solve the following math word problem.", "placement": "user",
                                  "variation_types": ["paraphrase_with_llm"]}
    assert cfg["prompt_format"] == {"text": "Question: {question}\n{arm_clause} {answer_format}",
                                    "variation_types": ["format structure"]}
    assert cfg["arm_clauses"]["zero_shot_cot"] == "Let's think step by step."
    assert cfg["variations_per_field"] == 8 and cfg["sampling_law"] == "uniform" and cfg["required_literals"] is None
    assert cfg["llm"]["platform"] == bb.REPLAY_PLATFORM and cfg["llm"]["model"] == "llama-3.3-70b-instruct-fp8"
    assert not set(bb.REFUSED_VARIATION_TYPES) & set(cfg["instruction"]["variation_types"] + cfg["prompt_format"]["variation_types"])
    assert "typos and noise" not in cfg["prompt_format"]["variation_types"]
    # the answer format is the one the format screening selected
    sel = json.loads((ROOT / "manifests" / "gsm8k_format_selection.json").read_text(encoding="utf-8"))
    assert cfg["answer_format"] == {"scorer_candidate": sel["registered"], "text": sel["text"]}
    assert cfg["answer_format"]["scorer_candidate"] == "boxed"
    assert bb.validate_config(cfg, False, CANDIDATES) == []


def test_a_pending_config_holds_no_active_choice():
    cfg = pending_cfg()
    with pytest.raises(bb.BankBuildError, match="PENDING_REVIEW"):
        bb.check_status(cfg, use_example=False)
    bb.check_status(cfg, use_example=True)                             # the example block may be built while pending
    problems = bb.validate_config(cfg, False, CANDIDATES)
    assert problems and all("None" in p or "must" in p for p in problems)


def test_example_block_is_labelled_complete_and_valid():
    cfg = load_config()
    assert bb.validate_config(cfg, True, CANDIDATES) == []
    text = CONFIG.read_text(encoding="utf-8")
    assert "EXAMPLE VALUES, not a decision" in text and text.index("EXAMPLE VALUES") < text.index("\nexample:")
    ex = cfg["example"]
    assert ex["llm"]["allowed"] is False and ex["answer_format"]["scorer_candidate"] in scoring.CANDIDATE_SPECS


@pytest.mark.parametrize("edit,needle", [
    (lambda b: b.update(seed=True), "seed"),
    (lambda b: b.update(seed=-1), "seed"),
    (lambda b: b.update(variations_per_field=0), "variations_per_field"),
    (lambda b: b.update(max_variations_per_row=0), "max_variations_per_row"),
    (lambda b: b.update(name="a/b"), "name"),
    (lambda b: b.update(name="gsm8k_bank"), "reserved"),
    (lambda b: b.update(sampling_law="weighted"), "sampling_law"),
    (lambda b: b.update(sampling_law=None), "sampling_law"),
    (lambda b: b.update(required_literals="Answer:"), "required_literals"),
    (lambda b: b.update(surprise=1), "unknown top-level key"),
    (lambda b: b.pop("seed"), "seed is missing"),
    (lambda b: b["instruction"].update(text="Solve. {answer_format}"), "{arm_clause}"),
    (lambda b: b["prompt_format"].update(text="Q: {question} {question}"), "2 times"),
    (lambda b: b["prompt_format"].update(text="Q:"), "{question}"),
    (lambda b: b["instruction"].update(variation_types=["context"]), "refused"),
    (lambda b: b["instruction"].update(variation_types=["shuffle"]), "refused"),
    (lambda b: b["instruction"].update(variation_types=["nonsense"]), "not supported"),
    (lambda b: b["instruction"].update(variation_types=None), "must be a list"),
    (lambda b: b["instruction"].update(variation_types=["format structure", "format structure"]), "twice"),
    (lambda b: b["instruction"].update(text="", variation_types=["format structure"]), "instruction.text is empty"),
    (lambda b: b["arm_clauses"].update(zero_shot_cot=b["arm_clauses"]["direct_answer"]), "no contrast"),
    (lambda b: b["arm_clauses"].update(direct_answer=None), "arm_clauses.direct_answer"),
    (lambda b: b["arm_clauses"].update(direct_answer="Answer {question}"), "placeholder"),
    (lambda b: b["answer_format"].update(text="Reply with {arm_clause}"), "placeholder"),
    (lambda b: b["answer_format"].update(scorer_candidate="nope"), "CANDIDATE_SPECS"),
    (lambda b: b["answer_format"].update(text="Give the number at the end."), "does not contain"),
    (lambda b: b["answer_format"].update(scorer_candidate="boxed"), "does not contain"),
    (lambda b: b["instruction"].update(variation_types=["paraphrase_with_llm"]), "llm.allowed is false"),
    (lambda b: (b["instruction"].update(variation_types=["paraphrase_with_llm"]), b["llm"].update(allowed=True)), "llm.platform"),
    (lambda b: (b["instruction"].update(variation_types=["paraphrase_with_llm"]),
                b["llm"].update(allowed=True, platform="TogetherAI")), "llm.model"),
    (lambda b: b["llm"].update(allowed="yes"), "llm.allowed"),
    (lambda b: b["instruction"].update(variation_types=[["format structure"]]), "not supported"),        # unhashable entries
    (lambda b: b["answer_format"].update(scorer_candidate=["answer_colon"]), "CANDIDATE_SPECS"),
    (lambda b: (b["instruction"].update(variation_types=["paraphrase_with_llm"]),
                b["llm"].update(allowed=True, platform=["TogetherAI"], model="m")), "llm.platform"),
])
def test_validate_config_reports_each_problem(edit, needle):
    cfg = confirmed_cfg()
    edit(cfg)
    problems = bb.validate_config(cfg, False, CANDIDATES)
    assert any(needle in p for p in problems), problems


def test_validate_config_accepts_a_complete_llm_configuration_without_calling_anything():
    cfg = confirmed_cfg()
    cfg["instruction"]["variation_types"] = ["paraphrase_with_llm", "format structure"]
    cfg["llm"] = {"allowed": True, "platform": "TogetherAI", "model": "some/model"}
    assert bb.validate_config(cfg, False, CANDIDATES) == []
    assert bb.llm_types_used(cfg) == ["paraphrase_with_llm"]


def test_make_template():
    block = example_block()
    t = bb.make_template(block)
    assert t == {"instruction": block["instruction"]["text"], "instruction variations": ["format structure"],
                 "prompt format": block["prompt_format"]["text"], "prompt format variations": ["format structure"]}
    block["instruction"] = {"text": "", "variation_types": []}
    block["prompt_format"]["variation_types"] = []
    assert bb.make_template(block) == {"prompt format": block["prompt_format"]["text"]}


# ---------------------------------------------------------------- LLM gate and environment

def test_llm_gate_requires_config_flag_and_environment_key():
    plain = example_block()
    assert bb.llm_gate(plain, False, {}) == []                         # no LLM type, nothing to gate
    block = example_block()
    block["instruction"]["variation_types"] = ["paraphrase_with_llm"]
    block["llm"] = {"allowed": True, "platform": "TogetherAI", "model": "m"}
    key = {"TOGETHER_API_KEY": "not-a-real-key"}
    assert bb.llm_gate(block, True, key) == []
    r = bb.llm_gate(block, False, key)
    assert len(r) == 1 and "--allow-llm-api" in r[0]
    r = bb.llm_gate(block, True, {})
    assert len(r) == 1 and "TOGETHER_API_KEY" in r[0]
    r = bb.llm_gate(block, True, {"OPENAI_API_KEY": "x"})                # a key for another platform does not count
    assert len(r) == 1 and "TOGETHER_API_KEY" in r[0]
    block["llm"]["allowed"] = False
    assert any("llm.allowed is false" in x for x in bb.llm_gate(block, True, key))
    block["llm"] = {"allowed": True, "platform": "Nowhere", "model": "m"}
    assert any("llm.platform" in x for x in bb.llm_gate(block, True, key))
    block["llm"] = {"allowed": True, "platform": ["TogetherAI"], "model": "m"}
    assert any("llm.platform" in x for x in bb.llm_gate(block, True, key))


def test_scrub_api_env_removes_every_key_variable():
    env = {"TOGETHER_API_KEY": "a", "OPENAI_API_KEY": "b", "ANTHROPIC_API_KEY": "c", "GOOGLE_API_KEY": "d",
           "COHERE_API_KEY": "e", "API_KEY": "f", "PATH": "p"}
    assert sorted(bb.scrub_api_env(env)) == sorted(["TOGETHER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                                                    "GOOGLE_API_KEY", "COHERE_API_KEY", "API_KEY"])
    assert env == {"PATH": "p"}
    assert bb.scrub_api_env(env) == []


def test_log_problems_flags_warnings_and_fallbacks_only():
    quiet = "✅ Loaded 3 rows from DataFrame\n\U0001f504 Pre-generating instruction variations (1 types)...\n" \
            "Creating row variations: 100%|##########| 16/16 [00:00<?, ?variation/s]\n"
    assert bb.log_problems(quiet) == []
    bad = quiet + "⚠️ Error generating paraphrase_with_llm variations: boom\n" \
                  "⚠️ Paraphrase augmenter requires api_key, using TextNoiseAugmenter as fallback\n"
    assert len(bb.log_problems(bad)) == 2


# ---------------------------------------------------------------- variants, skeletons, bank

def test_canonical_variants_and_hash_ignore_set_order():
    assert bb.canonical_variants({"b", "a", "orig", "c"}, "orig") == ["orig", "a", "b", "c"]
    assert bb.canonical_variants(["b", "a"], "missing") == ["a", "b"]
    assert bb.canonical_variants(["b", "a", "a"], None) == ["a", "b"]
    h1 = bb.variant_sha256(["a", "b", "a"], ["x", "y"])
    assert h1 == bb.variant_sha256(["b", "a"], ["y", "x"]) and h1 != bb.variant_sha256(["a", "b"], ["x", "z"])
    assert len(h1) == 64


def fab_block(**over):
    b = example_block()
    b.update(over)
    return b


def test_build_skeletons_dedups_drops_and_orders():
    block = fab_block()
    i0 = block["instruction"]["text"]                      # "Task: Solve the math problem. {arm_clause} {answer_format}"
    p0 = block["prompt_format"]["text"]                    # "Question: {question}\nResponse:"
    i_up = i0.replace("Task:", "TASK:")
    pairs = [
        (i_up, p0), (i0, p0), (i0, p0),                                     # a duplicated pair
        (i0, "question: {question}\nresponse:"),
        (i0.replace("{arm_clause}", ""), p0),                               # loses {arm_clause}
        (i0, p0 + " {question}"),                                           # duplicates {question}
        (i0.replace("{answer_format}", "{answer_formt}"), p0),              # mangled placeholder
    ]
    t = bb.build_skeletons(pairs, block)
    assert t["n_pairs"] == 6
    assert t["instruction_variants"][0] == i0 and t["instruction_variants"][1:] == sorted(t["instruction_variants"][1:])
    assert t["prompt_format_variants"][0] == p0
    kept = {(k["instruction_variant"], k["prompt_format_variant"]) for k in t["kept"]}
    iv, pv = t["instruction_variants"], t["prompt_format_variants"]
    assert kept == {(iv.index(i_up), pv.index(p0)), (iv.index(i0), pv.index(p0)),
                    (iv.index(i0), pv.index("question: {question}\nresponse:"))}
    assert [(k["instruction_variant"], k["prompt_format_variant"]) for k in t["kept"]] == sorted(kept)   # sorted by indices
    assert len(t["dropped"]) == 3 and all(d["reasons"] for d in t["dropped"])
    reasons = " | ".join(r for d in t["dropped"] for r in d["reasons"])
    assert "{arm_clause} occurs 0 times" in reasons and "{question} occurs 2 times" in reasons
    assert "{answer_format} occurs 0 times" in reasons
    k = t["kept"][0]
    assert set(k["messages"]) == set(bb.ARMS)
    d, c = k["messages"]["direct_answer"], k["messages"]["zero_shot_cot"]
    assert d[0]["content"].replace(block["arm_clauses"]["direct_answer"], "@") == \
        c[0]["content"].replace(block["arm_clauses"]["zero_shot_cot"], "@")      # the arms differ only in the clause
    assert d[1] == c[1]


def test_build_skeletons_drops_identical_messages_and_missing_literals():
    block = fab_block(required_literals=["Response:"])
    i0, p0 = block["instruction"]["text"], block["prompt_format"]["text"]
    pairs = [(i0, p0), (i0, p0 + "\n"),                                           # the user message is stripped: identical
             (i0, "Question {question}\nResponse")]                                 # loses the required literal
    t = bb.build_skeletons(pairs, block)
    assert len(t["kept"]) == 1
    reasons = [r for d in t["dropped"] for r in d["reasons"]]
    assert any("identical to those of" in r for r in reasons)
    assert any("required literal" in r and "Response:" in r for r in reasons)
    with pytest.raises(bb.BankBuildError):
        bb.assemble_bank(block, bb.build_skeletons([(i0.replace("{arm_clause}", ""), p0)], block),
                         generator_version="3.0.7", example=False, generation_record={})
    with pytest.raises(bb.BankBuildError):
        bb.build_skeletons([], block)


def test_assemble_bank_matches_the_d2_io_schema():
    block = fab_block()
    i0, p0 = block["instruction"]["text"], block["prompt_format"]["text"]
    pairs = [(i, p) for i in (i0, i0.replace("Task:", "TASK:"), i0.replace("Task:", "Task -"))
             for p in (p0, "QUESTION: {question}\nRESPONSE:")]
    t = bb.build_skeletons(pairs, block)
    bank = bb.assemble_bank(block, t, generator_version="3.0.7", example=False, generation_record={"note": "x"})
    assert bank["generator"] == "PromptSuite" and bank["generator_version"] == "3.0.7" and bank["seed"] == block["seed"]
    assert d2_io.validate_bank(bank) == []
    assert list(bank["configurations"]) == [f"ps-i{i:02d}-f{j:02d}" for i in range(3) for j in range(2)]
    probs = bank["sampling_probabilities"]
    assert set(probs) == set(bank["configurations"]) and all(abs(v - 1 / 6) < 1e-15 for v in probs.values())
    cfg = bank["configurations"]["ps-i00-f00"]
    assert cfg["dimensions"]["instruction_variant"] == 0 and cfg["dimensions"]["instruction_is_original"] is True
    assert cfg["dimensions"]["prompt_format_is_original"] is True
    assert cfg["dimensions"]["instruction_variation_types"] == ["format structure"]
    assert cfg["dimensions"]["answer_format_scorer_candidate"] == "answer_colon"
    assert bank["configurations"]["ps-i01-f01"]["dimensions"]["prompt_format_is_original"] is False
    assert bank["generation_record"] == {"note": "x"}
    ex = bb.assemble_bank(block, t, generator_version="3.0.7", example=True, generation_record={})
    assert bb.EXAMPLE_LABEL in ex["generator"] and ex["generator"].startswith("PromptSuite")
    assert d2_io.validate_bank(ex) == []


def test_assemble_bank_without_a_system_message_and_with_many_variants():
    block = fab_block()
    block["instruction"] = {"text": "", "variation_types": []}
    block["prompt_format"]["text"] = "{arm_clause} {answer_format}\nQ: {question}"
    block["prompt_format"]["variation_types"] = ["format structure"]
    pairs = [("", f"v{n}: {{arm_clause}} {{answer_format}}\nQ: {{question}}") for n in range(120)]
    t = bb.build_skeletons(pairs, block)
    bank = bb.assemble_bank(block, t, generator_version="3.0.7", example=False, generation_record={})
    assert d2_io.validate_bank(bank) == []
    assert len(bank["configurations"]) == 120 and "ps-i000-f000" in bank["configurations"]
    first = bank["configurations"]["ps-i000-f000"]["arms"]["direct_answer"]["messages"]
    assert [m["role"] for m in first] == ["user"]                         # no system message
    assert len({c.split("-f")[1] for c in bank["configurations"]}) == 120   # zero-padded, unique ids
    assert all(len(c.split("-f")[1]) == 3 for c in bank["configurations"])


def test_uniform_probabilities_sum_to_one():
    for n in (1, 3, 7, 16, 24, 1000):
        p = bb.uniform_probabilities([f"c{i}" for i in range(n)])
        assert abs(sum(p.values()) - 1) < 1e-9
    with pytest.raises(bb.BankBuildError):
        bb.uniform_probabilities([])


def test_canonical_bank_bytes_are_deterministic_utf8_lf():
    bank = {"a": "café\n", "b": [1, 2]}
    raw = bb.canonical_bank_bytes(bank)
    assert raw == bb.canonical_bank_bytes(copy.deepcopy(bank)) and raw.endswith(b"\n") and b"\r" not in raw
    assert "café" in raw.decode("utf-8")


def test_output_path_problem():
    assert bb.output_path_problem(ROOT / "manifests" / "gsm8k_bank.json", ROOT)
    assert bb.output_path_problem(ROOT / "manifests" / "x.json", ROOT)
    assert bb.output_path_problem(ROOT / "manifests", ROOT)
    assert bb.output_path_problem(ROOT / "data" / "bank_candidates" / "gsm8k_bank.json", ROOT)
    assert bb.output_path_problem(ROOT / "data" / "bank_candidates" / "example.json", ROOT) is None


# ---------------------------------------------------------------- script, in process (PromptSuite blocked)

@pytest.fixture
def blocked_promptsuite(monkeypatch):
    """Any import of promptsuite raises, so a passing refusal test proves the refusal came before the import."""
    monkeypatch.setitem(sys.modules, "promptsuite", None)
    for var in list(bb.PLATFORM_KEY_VARS.values()) + [bb.GENERIC_KEY_VAR]:
        monkeypatch.delenv(var, raising=False)
    return load_script()


def test_script_refuses_while_pending(blocked_promptsuite, tmp_path, capsys):
    path = write_cfg(tmp_path / "pending.yaml", pending_cfg())
    assert blocked_promptsuite.main(["--config", str(path), "--out-dir", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "PENDING_REVIEW" in err and "--example" in err
    assert not (tmp_path / "out").exists()


def test_script_refuses_the_shipped_config_before_the_format_is_registered(blocked_promptsuite, tmp_path, capsys):
    # the shipped config as it stood before the format screen selected boxed
    cfg = load_config()
    cfg["answer_format"] = {"scorer_candidate": None, "text": None}
    path = write_cfg(tmp_path / "c.yaml", cfg)
    assert blocked_promptsuite.main(["--config", str(path), "--out-dir", str(tmp_path / "out")]) == 2
    assert "answer_format" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_script_refuses_a_filled_config_that_is_still_pending(blocked_promptsuite, tmp_path, capsys):
    cfg = confirmed_cfg()
    cfg["status"] = "PENDING_REVIEW"
    path = write_cfg(tmp_path / "c.yaml", cfg)
    assert blocked_promptsuite.main(["--config", str(path), "--out-dir", str(tmp_path / "out")]) == 2
    assert "PENDING_REVIEW" in capsys.readouterr().err and not (tmp_path / "out").exists()


@pytest.mark.parametrize("allowed,flag,key,needle", [
    (True, False, True, "--allow-llm-api"),             # config allows it, flag missing
    (True, True, False, "TOGETHER_API_KEY"),            # flag given, no key in the environment
    (False, True, True, "llm.allowed"),                 # flag and key, config does not allow it
    (False, False, False, "llm.allowed"),
])
def test_script_refuses_hosted_llm_without_every_permission(blocked_promptsuite, tmp_path, capsys, monkeypatch,
                                                            allowed, flag, key, needle):
    cfg = confirmed_cfg()
    cfg["instruction"]["variation_types"] = ["paraphrase_with_llm"]
    cfg["llm"] = {"allowed": allowed, "platform": "TogetherAI", "model": "some/model"}
    path = write_cfg(tmp_path / "c.yaml", cfg)
    if key:
        monkeypatch.setenv("TOGETHER_API_KEY", "not-a-real-key")
    out = tmp_path / "out"
    args = ["--config", str(path), "--out-dir", str(out)] + (["--allow-llm-api"] if flag else [])
    assert blocked_promptsuite.main(args) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and needle in err
    assert not out.exists()                                             # exit 2 (refused), not 1: the blocked import was never reached


def test_script_refuses_an_unusable_variation_type_and_unknown_candidate(blocked_promptsuite, tmp_path, capsys):
    cfg = confirmed_cfg()
    cfg["prompt_format"]["variation_types"] = ["context"]
    cfg["answer_format"]["scorer_candidate"] = "nope"
    path = write_cfg(tmp_path / "c.yaml", cfg)
    assert blocked_promptsuite.main(["--config", str(path), "--out-dir", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "context" in err and "CANDIDATE_SPECS" in err


def test_script_never_writes_under_manifests(blocked_promptsuite, capsys):
    before = sorted(p.name for p in (ROOT / "manifests").iterdir())
    assert blocked_promptsuite.main(["--example", "--out-dir", str(ROOT / "manifests")]) == 2
    assert "manifests" in capsys.readouterr().err
    assert sorted(p.name for p in (ROOT / "manifests").iterdir()) == before


def test_script_refuses_without_the_fixed_hash_seed_in_process(tmp_path, monkeypatch, capsys):
    """Called in process under a randomized hash seed, the generation step refuses instead of producing a bank."""
    if sys.flags.hash_randomization == 0:
        pytest.skip("this interpreter already runs with PYTHONHASHSEED=0")
    monkeypatch.setitem(sys.modules, "promptsuite", None)
    for var in list(bb.PLATFORM_KEY_VARS.values()) + [bb.GENERIC_KEY_VAR]:
        monkeypatch.delenv(var, raising=False)                  # the script scrubs these; monkeypatch restores them
    mod = load_script()
    assert mod.main(["--example", "--out-dir", str(tmp_path)]) == 2
    assert "PYTHONHASHSEED" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- build_requests dry run

def _models_yaml(path):
    entry = {"family": "test", "hf_id": "org/model", "revision": "a" * 40, "dtype": "bfloat16", "quantization": "none",
             "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": 1,
             "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
             "thinking_off": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False},
                              "system_prompt_token": None, "rationale": None}, "kernels": "batch_invariant"}
    path.write_text(yaml.safe_dump({"dataset2": {"models": {"_template": {"hf_id": "PLACEHOLDER"}, "m1": entry}}}))
    return path


def _decoding_yaml(path):
    path.write_text(yaml.safe_dump({
        "status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
        "max_new_tokens": {"direct_answer": 400, "zero_shot_cot": 400}, "stop": [],
        "include_stop_str_in_output": False, "skip_special_tokens": True,
        "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                   "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}))
    return path


def dry_run_build_requests(bank_path, tmp_path, capsys, extra):
    items = pd.DataFrame({"item_id": pd.Series(range(12), dtype="int64"),
                          "question": [f"What is {i}+{i}? (use {{braces}} freely)" for i in range(12)],
                          "gold_answer": [str(2 * i) for i in range(12)]})
    items_path = tmp_path / "items.parquet"
    items.to_parquet(items_path, index=False)
    spec = importlib.util.spec_from_file_location("build_requests_cli", ROOT / "scripts" / "build_requests.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = ["--bank", str(bank_path), "--models", "m1", "--allow-unregistered", "--items-parquet", str(items_path),
            "--models-yaml", str(_models_yaml(tmp_path / "models.yaml")),
            "--decoding-yaml", str(_decoding_yaml(tmp_path / "decoding.yaml")), "--out-dir", str(tmp_path / "req"),
            "--split", "train", "--calibration-items", "5", "--calibration-manifest", str(tmp_path / "cal.json"), *extra]
    capsys.readouterr()
    code = cli.main(args)
    cap = capsys.readouterr()
    return code, cap, cli


def check_requests(tmp_path, bank, n_cfg, kind_dir):
    record = json.loads((tmp_path / "req" / kind_dir / "build_record.json").read_text(encoding="utf-8"))
    assert record["split"] == "train" and record["excluded_from_main"] is True
    assert record["bank"]["generator"] == bank["generator"] and len(record["configuration_ids"]) == n_cfg
    df = pd.read_parquet(tmp_path / "req" / kind_dir / "m1.parquet")
    assert len(df) == n_cfg * 2 * 5
    for _, row in df.iterrows():
        msgs = json.loads(row["messages_json"])
        text = "\n".join(m["content"] for m in msgs)
        assert "{arm_clause}" not in text and "{answer_format}" not in text and msgs[-1]["role"] == "user"
        assert text.count("What is ") == 1                      # the question was substituted exactly once


def test_build_requests_dry_run_accepts_a_fabricated_candidate(tmp_path, capsys):
    block = fab_block()
    i0, p0 = block["instruction"]["text"], block["prompt_format"]["text"]
    pairs = [(i, p) for i in (i0, i0.replace("Task:", "TASK:")) for p in (p0, "QUESTION: {question}\nRESPONSE:")]
    bank = bb.assemble_bank(block, bb.build_skeletons(pairs, block), generator_version="3.0.7", example=True,
                            generation_record={})
    bank_path = tmp_path / "bank.json"
    bank_path.write_bytes(bb.canonical_bank_bytes(bank))
    code, cap, _ = dry_run_build_requests(bank_path, tmp_path, capsys,
                                          ["--tranche", "--configs", "ps-i00-f00", "ps-i01-f01"])
    assert code == 0, cap.err
    check_requests(tmp_path, bank, 2, "tranche")
    code, cap, _ = dry_run_build_requests(bank_path, tmp_path, capsys, ["--screening"])
    assert code == 0, cap.err
    check_requests(tmp_path, bank, 4, "screening")


# ---------------------------------------------------------------- PromptSuite, through .venv-promptsuite

@pytest.fixture(scope="module")
def example_build(tmp_path_factory):
    if not VENV_PY.exists():
        pytest.skip(".venv-promptsuite is not set up")
    out = tmp_path_factory.mktemp("bank_a")
    cwd = tmp_path_factory.mktemp("cwd_a")
    proc = run_builder(["--example", "--out-dir", out], cwd, PYTHONHASHSEED="3")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    path = Path(summary["path"])
    return {"proc": proc, "summary": summary, "path": path, "bank": json.loads(path.read_text(encoding="utf-8")),
            "raw": path.read_bytes(), "dir": out}


@needs_venv
def test_example_build_is_a_valid_labelled_bank(example_build):
    s, bank = example_build["summary"], example_build["bank"]
    assert s["status"] == "written" and s["n_configurations"] == 16 and s["n_dropped"] == 0 and s["llm_used"] is False
    assert s["pythonhashseed"] == "0"                          # the script re-executed itself under a fixed hash seed
    assert bb.EXAMPLE_LABEL in bank["generator"] and bank["generator"].startswith("PromptSuite")
    assert example_build["path"].name == "example.json" and example_build["path"].parent == example_build["dir"]
    assert d2_io.validate_bank(bank) == []
    assert bank["generator_version"] == "3.0.7" and bank["seed"] == 1234
    assert len(bank["configurations"]) == 16 and abs(sum(bank["sampling_probabilities"].values()) - 1) < 1e-9
    # round trip through the project loader (duplicate-key check, hashes)
    loaded, sha = d2_io.load_bank(example_build["path"])
    assert sha == hashlib.sha256(example_build["raw"]).hexdigest() == s["sha256"] and loaded == bank


@needs_venv
def test_example_build_arms_differ_only_in_the_clause_and_every_item_gets_one_skeleton(example_build):
    bank = example_build["bank"]
    rec = bank["generation_record"]
    clauses, fmt = rec["arm_clauses"], rec["answer_format"]["text"]
    for cid, cfg in bank["configurations"].items():
        d, c = (cfg["arms"][a]["messages"] for a in bb.ARMS)
        assert [m["role"] for m in d] == ["system", "user"] == [m["role"] for m in c]
        assert d[0]["content"].count(clauses["direct_answer"]) == 1 and c[0]["content"].count(clauses["zero_shot_cot"]) == 1
        assert fmt in d[0]["content"] and fmt in c[0]["content"]
        assert d[0]["content"].replace(clauses["direct_answer"], "@") == c[0]["content"].replace(clauses["zero_shot_cot"], "@")
        assert d[1] == c[1] and d[1]["content"].count("{question}") == 1
        assert "{question}" not in d[0]["content"]
        for m in d + c:
            assert "{arm_clause}" not in m["content"] and "{answer_format}" not in m["content"]
        dims = cfg["dimensions"]
        assert 0 <= dims["instruction_variant"] < 4 and 0 <= dims["prompt_format_variant"] < 4
        assert dims["instruction_variation_types"] == ["format structure"]


@needs_venv
def test_example_build_generation_record(example_build):
    rec = example_build["bank"]["generation_record"]
    assert rec["promptsuite_version"] == "3.0.7" and rec["pythonhashseed"] == "0" and rec["config_status"] == "EXAMPLE"
    assert rec["seed"] == 1234 and rec["variations_per_field"] == 4 and rec["max_variations_per_row"] is None
    assert rec["template"]["instruction variations"] == ["format structure"]
    assert rec["llm"] == {"used": False, "platform": None, "model": None}
    assert rec["variant_sha256"] == bb.variant_sha256(rec["variants"]["instruction"], rec["variants"]["prompt_format"])
    assert rec["variants"]["instruction"][0] == rec["template"]["instruction"]             # the original is variant 0
    assert rec["variants"]["prompt_format"][0] == rec["template"]["prompt format"]
    assert rec["fidelity_check"]["result"] == "identical" and rec["fidelity_check"]["messages_compared"] == 32
    assert rec["answer_format"]["scorer_candidate"] == "answer_colon"
    assert rec["answer_format"]["scorer_version"] == scoring.SCORER_VERSION
    assert rec["dropped"] == [] and rec["warnings"] == [] and rec["n_skeleton_pairs"] == 16 == rec["n_configurations"]
    assert rec["sampling_law"] == "uniform" and rec["python_version"].startswith("3.")


@needs_venv
def test_example_build_is_byte_identical_across_runs_and_starting_hash_seeds(example_build, tmp_path):
    out2 = tmp_path / "again"
    proc = run_builder(["--example", "--out-dir", out2], tmp_path, PYTHONHASHSEED="11")
    assert proc.returncode == 0, proc.stderr
    assert (out2 / "example.json").read_bytes() == example_build["raw"]
    # the same directory again leaves the file alone
    proc = run_builder(["--example", "--out-dir", example_build["dir"]], tmp_path)
    assert proc.returncode == 0 and json.loads(proc.stdout)["status"] == "unchanged"


@needs_venv
def test_hash_seed_dependent_configuration_is_still_byte_identical(tmp_path):
    """Two types in one field make the kept variants depend on string hashing unless the script fixes the seed."""
    cfg = confirmed_cfg(name="multi", variations_per_field=6)
    cfg["instruction"]["text"] = "Task: Solve the math problem carefully. {arm_clause} {answer_format}"
    cfg["instruction"]["variation_types"] = ["typos and noise", "format structure"]
    cfg["prompt_format"]["variation_types"] = ["format structure"]
    path = write_cfg(tmp_path / "multi.yaml", cfg)
    raws = []
    for n, hs in enumerate(("1", "7")):
        proc = run_builder(["--config", path, "--out-dir", tmp_path / f"o{n}"], tmp_path, PYTHONHASHSEED=hs)
        assert proc.returncode == 0, proc.stderr
        raws.append((tmp_path / f"o{n}" / "multi.json").read_bytes())
    assert raws[0] == raws[1]
    bank = json.loads(raws[0].decode("utf-8"))
    assert d2_io.validate_bank(bank) == [] and bank["generator"] == "PromptSuite"          # a confirmed build is not labelled
    assert bank["generation_record"]["config_status"] == "confirmed"
    assert any("several variation types" in w for w in bank["generation_record"]["warnings"])


@needs_venv
def test_environment_matches_the_pins_in_requirements_promptsuite():
    def norm(line):
        name, _, ver = line.strip().partition("==")
        return name.lower().replace("_", "-"), ver
    pins = {norm(ln) for ln in (ROOT / "requirements-promptsuite.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}
    proc = subprocess.run([str(VENV_PY), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    frozen = {norm(ln) for ln in proc.stdout.splitlines() if ln.strip() and "==" in ln}
    assert {x for x in frozen if x[0] != "pip"} == pins
    assert ("promptsuite", "3.0.7") in pins and bb.PROMPTSUITE_VERSION == "3.0.7"


@needs_venv
def test_capped_and_system_free_configurations_build(tmp_path):
    capped = confirmed_cfg(name="capped", max_variations_per_row=5)
    nosys = confirmed_cfg(name="nosys")
    nosys["instruction"] = {"text": "", "variation_types": []}
    nosys["prompt_format"]["text"] = "Question: {question}\n{arm_clause} {answer_format}\nResponse:"
    for cfg in (capped, nosys):
        proc = run_builder(["--config", write_cfg(tmp_path / f"{cfg['name']}.yaml", cfg), "--out-dir", tmp_path / "o"], tmp_path)
        assert proc.returncode == 0, proc.stderr
    cap = json.loads((tmp_path / "o" / "capped.json").read_text(encoding="utf-8"))
    assert d2_io.validate_bank(cap) == [] and len(cap["configurations"]) == 5          # PromptSuite kept 5 of 16 combinations
    assert any("max_variations_per_row is 5" in w for w in cap["generation_record"]["warnings"])
    sysfree = json.loads((tmp_path / "o" / "nosys.json").read_text(encoding="utf-8"))
    assert d2_io.validate_bank(sysfree) == [] and len(sysfree["configurations"]) == 4  # 1 instruction variant x 4
    for c in sysfree["configurations"].values():
        for a in bb.ARMS:
            assert [m["role"] for m in c["arms"][a]["messages"]] == ["user"]
    assert sysfree["generation_record"]["fidelity_check"]["result"] == "identical"
    assert sysfree["generation_record"]["warnings"] == []


@needs_venv
def test_cli_refuses_pending_config_through_the_venv(tmp_path):
    out = tmp_path / "o"
    proc = run_builder(["--config", write_cfg(tmp_path / "pending.yaml", pending_cfg()), "--out-dir", out], tmp_path)
    assert proc.returncode == 2 and "PENDING_REVIEW" in proc.stderr and not out.exists()


@needs_venv
def test_cli_refuses_hosted_llm_through_the_venv_without_calling_anything(tmp_path):
    cfg = confirmed_cfg(name="llm")
    cfg["instruction"]["variation_types"] = ["paraphrase_with_llm"]
    cfg["llm"] = {"allowed": True, "platform": "TogetherAI", "model": "some/model"}
    path = write_cfg(tmp_path / "llm.yaml", cfg)
    out = tmp_path / "o"
    proc = run_builder(["--config", path, "--out-dir", out], tmp_path)                       # no flag, no key
    assert proc.returncode == 2 and "--allow-llm-api" in proc.stderr
    proc = run_builder(["--config", path, "--out-dir", out, "--allow-llm-api"], tmp_path)    # flag, still no key
    assert proc.returncode == 2 and "TOGETHER_API_KEY" in proc.stderr
    assert not out.exists()


@needs_venv
def test_cli_overwrite_guard(example_build, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "example.json").write_text("{}", encoding="utf-8")
    proc = run_builder(["--example", "--out-dir", out], tmp_path)
    assert proc.returncode == 2 and "--overwrite" in proc.stderr and (out / "example.json").read_text() == "{}"
    proc = run_builder(["--example", "--out-dir", out, "--overwrite"], tmp_path)
    assert proc.returncode == 0 and (out / "example.json").read_bytes() == example_build["raw"]


@needs_venv
def test_dotenv_files_above_the_environment_are_not_read(tmp_path):
    """PromptSuite calls load_dotenv() at import, and python-dotenv searches upward from the package directory,
    which sits inside the repository. A .env file above the environment must not reach the process."""
    canary = ROOT / ".venv-promptsuite" / ".env"
    if canary.exists():
        pytest.skip("a .env file already exists in .venv-promptsuite; the test will not touch it")
    unguarded = tmp_path / "probe_unguarded.py"
    unguarded.write_text("import os\nimport promptsuite\nprint(os.environ.get('BANK_BUILDER_CANARY'))\n", encoding="utf-8")
    guarded = tmp_path / "probe_guarded.py"
    guarded.write_text(
        "import importlib.util, os\n"
        f"spec = importlib.util.spec_from_file_location('b', r'{SCRIPT}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"import yaml; cfg = yaml.safe_load(open(r'{CONFIG}', encoding='utf-8'))\n"
        "block = cfg['example']\n"
        "from src import bank_build as bb\n"
        "m.run_promptsuite(block, bb.make_template(block), None)\n"
        "print(os.environ.get('BANK_BUILDER_CANARY'))\n", encoding="utf-8")
    # script files, not -c: in -c mode python-dotenv searches from the working directory instead of the package
    env = clean_env(PYTHONHASHSEED="0", PYTHONPATH=str(ROOT))
    try:
        canary.write_text("BANK_BUILDER_CANARY=leaked\n", encoding="utf-8")
        plain = subprocess.run([str(VENV_PY), str(unguarded)], cwd=str(tmp_path), env=env, capture_output=True,
                               text=True, timeout=120)
        guard = subprocess.run([str(VENV_PY), str(guarded)], cwd=str(tmp_path), env=env, capture_output=True,
                               text=True, timeout=120)
    finally:
        canary.unlink(missing_ok=True)
    assert plain.stdout.strip() == "leaked", plain.stderr            # the hazard is real without the guard
    assert guard.returncode == 0, guard.stderr
    assert guard.stdout.strip() == "None"                            # and the builder's guard stops it


@needs_venv
def test_build_requests_dry_run_accepts_the_example_candidate(example_build, tmp_path, capsys):
    bank = example_build["bank"]
    ids = list(bank["configurations"])
    code, cap, _ = dry_run_build_requests(example_build["path"], tmp_path, capsys,
                                          ["--tranche", "--configs", ids[0], ids[5], ids[-1]])
    assert code == 0, cap.err
    check_requests(tmp_path, bank, 3, "tranche")
    assert json.loads((tmp_path / "req" / "tranche" / "build_record.json").read_text())["bank"]["generator"] \
        .endswith("EXAMPLE, not a registered bank)")
    code, cap, _ = dry_run_build_requests(example_build["path"], tmp_path, capsys, ["--screening"])
    assert code == 0, cap.err
    check_requests(tmp_path, bank, 16, "screening")


# ---------------------------------------------------------------- replayed paraphrase (the shipped design)

PARAPHRASES = ["Work out the math word problem below.", "Find the answer to this math word problem.",
               "Please solve the math word problem that follows.", "Calculate the solution to the following word problem.",
               "Answer the math word problem given below.", "Determine the solution to this math word problem.",
               "Solve this word problem involving math."]


def replay_inputs(tmp_path, prompt_sha=None, drop=None):
    """The shipped configuration with an answer format, a fabricated record of the paraphraser's answer and a review."""
    import importlib.util as iu
    spec = iu.spec_from_file_location("paraphrase_d2", ROOT / "scripts" / "paraphrase_d2.py")
    pd2 = iu.module_from_spec(spec)
    spec.loader.exec_module(pd2)
    cfg = load_config()
    cfg["answer_format"] = {"scorer_candidate": "answer_colon_markup",
                            "text": 'End your reply with a final line of the form "Answer: <number>".'}
    cfg["llm"]["record"], cfg["llm"]["review"] = str(tmp_path / "record.json"), str(tmp_path / "review.json")
    path = write_cfg(tmp_path / "c.yaml", cfg)
    assert pd2.main(["--config", str(path), "request", "--out-dir", str(tmp_path / "req")]) == 0
    req = json.loads((tmp_path / "req" / "request.json").read_text(encoding="utf-8"))
    answer = repr(PARAPHRASES)
    record = {"kind": bb.RECORD_KIND, "request": {"prompt_sha256": prompt_sha or req["prompt_sha256"]},
              "response": {"text": answer, "sha256": bb.sha256_text(answer), "finish_reason": "stop"},
              "replay": {"text": answer, "sha256": bb.sha256_text(answer), "code_fence_removed": False},
              "model": {"key": cfg["llm"]["model"]}}
    raw = json.dumps(record).encode("utf-8")
    (tmp_path / "record.json").write_bytes(raw)
    review = {"kind": bb.REVIEW_KIND, "record_sha256": hashlib.sha256(raw).hexdigest(), "reviewers": ["Test"],
              "verdicts": [{"text": x, "verdict": "drop" if x == drop else "keep",
                            "note": "changes the task" if x == drop else ""} for x in PARAPHRASES]}
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    return path, req


@needs_venv
def test_replayed_paraphrase_builds_the_shipped_design_in_the_user_turn(tmp_path):
    path, req = replay_inputs(tmp_path, drop=PARAPHRASES[6])
    assert req["n_augments"] == 7 and "Solve the following math word problem." in req["messages"][0]["content"]
    proc = run_builder(["--config", path, "--out-dir", tmp_path / "out"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    bank = json.loads(Path(json.loads(proc.stdout)["path"]).read_text(encoding="utf-8"))
    rec = bank["generation_record"]
    assert rec["llm"]["replay"]["dropped_in_review"] == {PARAPHRASES[6]: "changes the task"}
    assert rec["instruction_placement"] == "user" and rec["fidelity_check"]["compared_with"] == "prompt"
    assert set(rec["variants"]["instruction"]) == {"Solve the following math word problem.", *PARAPHRASES}
    reasons = [r for d in rec["dropped"] for r in d["reasons"]]
    assert any("dropped in the meaning review: changes the task" in r for r in reasons)
    for c in bank["configurations"].values():
        msgs = {a: c["arms"][a]["messages"] for a in ARMS_}
        assert all(len(m) == 1 and m[0]["role"] == "user" for m in msgs.values())
        d, z = msgs["direct_answer"][0]["content"], msgs["zero_shot_cot"][0]["content"]
        assert d.count("{question}") == 1 and "Let's think step by step." in z and z.replace(
            "Let's think step by step.", "Give only the final answer, without any working.") == d
        assert PARAPHRASES[6] not in d
    assert len(bank["configurations"]) <= 7 * 8


@needs_venv
def test_a_record_for_another_request_is_refused(tmp_path):
    path, _ = replay_inputs(tmp_path, prompt_sha="0" * 64)
    proc = run_builder(["--config", path, "--out-dir", tmp_path / "out"], tmp_path)
    assert proc.returncode == 2 and "the record answers" in proc.stderr and not (tmp_path / "out").exists()


ARMS_ = ("direct_answer", "zero_shot_cot")
