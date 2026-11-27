"""Tests for MATH-500 in the Dataset 2 pipeline (src/math_d2.py, src/string_tasks.py and the --task math paths of
scripts/build_requests.py, score_d2.py, screen_d2.py, freeze_bank_d2.py, draw_tranche_d2.py, freeze_sampler_d2.py,
run_dataset2.py, precision_matched.py, build_forced_direct.py and score_forced_direct.py).

The chain test runs on fabricated item tables, a fabricated items manifest, a fabricated candidate bank and four
fabricated models in two registries, with the real runner and an echo backend whose outputs are fabricated. Tests
on the pinned item tables skip when they are absent.
"""
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import bbh_d2, d2_io, generation as gen, math_d2, math_task as mt, screening, string_tasks

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_math_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_cli = load_script("build_requests")
score_cli = load_script("score_d2")
screen_cli = load_script("screen_d2")
freeze_cli = load_script("freeze_bank_d2")
draw_cli = load_script("draw_tranche_d2")
sampler_cli = load_script("freeze_sampler_d2")
rd2 = load_script("run_dataset2")
forced_cli = load_script("build_forced_direct")
sfd_cli = load_script("score_forced_direct")


# ---------------------------------------------------------------- fabricated inputs

GOLDS = ["12", "-3", "1/2", "1000", "-7/4", "0"]
SET_INDICES = {"screening": (0, 6), "calibration": (6, 12), "target": (12, 18)}


def item_rows(item_set):
    lo, hi = SET_INDICES[item_set]
    pre = "test/algebra/{}.json" if item_set == "target" else "train/algebra/{}"
    rows = [{"item_id": pre.format(i), "question": f"Compute the value asked in problem number {i}.",
             "gold_answer": GOLDS[i % len(GOLDS)], "subject": "Algebra", "level": f"Level {1 + i % 5}"}
            for i in range(lo, hi)]
    return pd.DataFrame(rows, columns=list(mt.TABLE_COLUMNS))


def write_items(tmp):
    man = {"task": "math", "target_revision": mt.REV_TARGET, "train_revision": mt.REV_TRAIN, "tables": {}, "sets": {}}
    for name in SET_INDICES:
        df = item_rows(name)
        raw = mt.table_bytes(df)
        p = tmp / f"items_{name}.parquet"
        p.write_bytes(raw)
        man["tables"][name] = {"path": str(p), "n_rows": len(df), "sha256": hashlib.sha256(raw).hexdigest()}
        man["sets"][name] = {"n_items": len(df), "canonical_sha256": mt.canonical_sha256(df)}
    path = tmp / "math_items.json"
    path.write_text(json.dumps(man), encoding="utf-8")
    return path, man


def make_candidate():
    def cfg(instr):
        def arm(clause):
            return {"messages": [{"role": "user", "content": f"{instr}\n\nQuestion: {{question}}\n{clause} End your reply "
                                                             "with a final line that contains \\boxed{<number>}."}]}
        return {"dimensions": {"instruction": instr},
                "arms": {"direct_answer": arm("Give only the final answer, without any working."),
                         "zero_shot_cot": arm("Let's think step by step.")}}
    ids = ["c1", "c2", "c3"]
    return {"generator": "handwritten-math-test", "generator_version": "0", "seed": 20261103,
            "generation_record": {"config_status": "confirmed", "sampling_law": "uniform"},
            "sampling_probabilities": {c: 1 / 3 for c in ids},
            "configurations": {"c1": cfg("Solve the following math problem."), "c2": cfg("Find the answer."),
                               "c3": cfg("NOBOX Work out the problem.")}}


def model_entry(**over):
    e = {"family": "test", "hf_id": "org/model", "revision": SHA, "dtype": "bfloat16", "quantization": "none",
         "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": 1,
         "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
         "thinking_off": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False},
                          "system_prompt_token": None, "rationale": None}, "kernels": "batch_invariant"}
    e.update(over)
    return e


def decoding_dict():
    return {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
            "max_new_tokens": {"direct_answer": 400, "zero_shot_cot": 400}, "stop": [],
            "include_stop_str_in_output": False, "skip_special_tokens": True,
            "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                       "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}


class MathEcho(gen.EchoBackend):
    """Echo backend with fabricated MATH answers: a final \\boxed{} in LaTeX form (none for configuration c3, whose
    instruction carries NOBOX), correct for about half of the prompts, and long direct answers for model m3."""

    def __init__(self, spec, decoding, items):
        super().__init__(spec, decoding)
        self.items = items

    def generate(self, conversations, params):
        res = super().generate(conversations, params)
        out = []
        for conv, r in zip(conversations, res):
            user = conv[-1]["content"]
            q = max((q for q in self.items if q in user), key=len)
            gold = self.items[q]
            ok = hashlib.sha256((r.prompt_echo + self.spec.model_key).encode("utf-8")).digest()[0] % 2 == 0
            fr = Fraction(gold) + (0 if ok else 1)
            ans = mt.latex_form(str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}")
            direct = "Give only the final answer" in user
            body = "word " * 200 if (self.spec.model_key == "m3" and direct) else ("" if direct else "Some steps.\n")
            text = body + ("The answer follows." if "NOBOX" in user else f"\\boxed{{{ans}}}")
            out.append(gen.GenResult(text, r.n_prompt_tokens, len(text.split()), "stop", r.prompt_echo))
        return out


class Env:
    def __init__(self, tmp, monkeypatch):
        self.tmp = tmp
        self.items_manifest, self.man = write_items(tmp)
        monkeypatch.setattr(math_d2, "ITEMS_MANIFEST", self.items_manifest)
        self.cand_path = tmp / "candidate.json"
        self.cand_path.write_text(json.dumps(make_candidate()), encoding="utf-8")
        self.reg1, self.reg2 = tmp / "models.yaml", tmp / "models_extension.yaml"
        self.reg1.write_text(yaml.safe_dump({"dataset2": {"models": {
            "_template": {"hf_id": "PLACEHOLDER"}, "m1": model_entry(family="Alpha", hf_id="org/m1"),
            "m2": model_entry(family="Beta", hf_id="org/m2")}}}))
        self.reg2.write_text(yaml.safe_dump({"dataset2": {"models": {
            "m3": model_entry(family="Beta", hf_id="org/m3"), "m4": model_entry(family="Alpha", hf_id="org/m4"),
            "m5": model_entry(family="Alpha", hf_id="org/m5")}}}))
        monkeypatch.setattr(bbh_d2, "REGISTRIES", (self.reg1, self.reg2))
        panel = {"panel": [{"repo": f"org/m{i}", "tier": t} for i, t in ((1, "low"), (2, "middle"), (3, "high"),
                                                                         (4, "high"), (5, "low"))]}
        self.panel_path = tmp / "panel.json"
        self.panel_path.write_text(json.dumps(panel), encoding="utf-8")
        monkeypatch.setattr(bbh_d2, "PANEL_EXTENSION", self.panel_path)
        self.gsm8k29 = tmp / "gsm8k30.json"                       # the GSM8K panel: m5 is not in it
        self.gsm8k29.write_text(json.dumps({"models": ["m1", "m2", "m3", "m4"]}), encoding="utf-8")
        monkeypatch.setattr(math_d2, "PANEL_SOURCE", self.gsm8k29)
        self.decoding = tmp / "decoding.yaml"
        self.decoding.write_text(yaml.safe_dump(decoding_dict()))
        self.req = tmp / "requests"
        self.questions = {}
        for name in SET_INDICES:
            df = item_rows(name)
            self.questions.update(dict(zip(df["question"], df["gold_answer"])))

    def build(self, *extra):
        return build_cli.main(["--task", "math", "--out-dir", str(self.req), "--decoding-yaml", str(self.decoding),
                               *map(str, extra)])

    def generate(self, key, sub):
        reg = Path(math_d2.registry_for(key))
        reg = reg if reg.is_absolute() else ROOT / reg
        spec = gen.parse_model_spec(key, gen.load_registry(reg))
        dec = gen.load_decoding(self.decoding)
        out = self.tmp / "gen" / (sub or "main") / key
        req = self.req / sub / f"{key}.parquet" if sub else self.req / f"{key}.parquet"
        log = gen.run_generation(req, key, out, backend="echo", backend_obj=MathEcho(spec, dec, self.questions),
                                 models_path=reg, decoding_path=self.decoding, verbose=False, shard_size=1000)
        assert log["status"] == "complete"
        return out

    def score(self, dirs, out, *extra):
        return score_cli.main(["--task", "math", "--work-dir", *map(str, dirs), "--requests-dir", str(self.req),
                               "--out", str(out), "--allow-test-backend", *extra])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "gpu_info", lambda timeout=10.0: {"count": 0, "devices": [], "driver_version": None,
                                                              "cuda_version_driver": None, "nvidia_smi": False})
    return Env(tmp_path, monkeypatch)


# ---------------------------------------------------------------- modules

def test_string_tasks_map():
    assert string_tasks.TASKS == ("gsm8k", "bbh", "math", "hotpot")      # HotpotQA is the fourth task
    assert string_tasks.module("math") is math_d2 and string_tasks.module("bbh") is bbh_d2
    assert string_tasks.module("gsm8k") is None
    for tm in (bbh_d2, math_d2):
        for name in ("TASK", "TITLE", "SCORED", "REQUESTS_DIR", "KIND_ITEM_SET", "KIND_SPLIT", "FIXES_ITEM_SETS",
                     "FIXES_SCORER", "SCREEN_PURPOSE", "SCREEN_METRICS", "SCORER_VERSION", "TARGET_LABEL",
                     "load_items", "load_scorer", "score_fn", "row_metrics", "answer_types", "load_registries",
                     "registry_for", "extension_tiers"):
            assert hasattr(tm, name), (tm.TASK, name)
    assert math_d2.KIND_ITEM_SET == bbh_d2.KIND_ITEM_SET and math_d2.KIND_SPLIT == bbh_d2.KIND_SPLIT


def test_load_items_checks_the_manifest(env, tmp_path):
    for name in SET_INDICES:
        df, info = math_d2.load_items(name)
        assert len(df) == 6 and list(df.columns) == list(math_d2.ITEM_COLUMNS)
        assert info["item_set"] == name and info["split"] == ("test" if name == "target" else "train")
        assert info["canonical_sha256"] == env.man["sets"][name]["canonical_sha256"] and info["matches_manifest"]
    p = Path(env.man["tables"]["target"]["path"])
    df = pd.read_parquet(p)
    df.loc[0, "gold_answer"] = "13"
    df.to_parquet(p, index=False)
    with pytest.raises(math_d2.MathPipelineError, match="differs"):
        math_d2.load_items("target")
    with pytest.raises(math_d2.MathPipelineError, match="item set"):
        math_d2.load_items("train")
    custom = tmp_path / "custom.parquet"
    item_rows("target").to_parquet(custom, index=False)
    df, info = math_d2.load_items("target", custom)
    assert info["matches_manifest"] is None and "dry run" in info["source"]
    item_rows("target").assign(gold_answer="0.5").to_parquet(custom, index=False)
    with pytest.raises(math_d2.MathPipelineError, match="canonical number"):
        math_d2.load_items("target", custom)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**env.man, "train_revision": "b" * 40}), encoding="utf-8")
    with pytest.raises(math_d2.MathPipelineError, match="pinned MATH revisions"):
        math_d2.load_items("target", manifest_path=bad)


def test_load_scorer(tmp_path):
    rec = math_d2.load_scorer()
    assert rec["status"] == "registered" and rec["scorer_version"] == mt.SCORER_VERSION
    p = tmp_path / "scorer.json"
    p.write_text(json.dumps({**mt.scorer_record(), "scorer_version": "math-boxed/0.9"}), encoding="utf-8")
    with pytest.raises(math_d2.MathPipelineError, match="differs"):
        math_d2.load_scorer(p)


def test_panel_is_the_gsm8k29_panel(env, tmp_path):
    assert math_d2.panel() == ["m1", "m2", "m3", "m4"]
    with pytest.raises(math_d2.MathPipelineError, match="GSM8K-29 panel"):
        math_d2.panel(tmp_path / "absent.json")


def test_math_tranche_config_is_fixed():
    t = yaml.safe_load((ROOT / "config" / "tranche_math.yaml").read_text(encoding="utf-8"))
    assert t == {"status": "confirmed", "configurations": 8, "seed": 20261103, "calibration_items": 398}
    assert draw_cli.STRING_TASK["math"]["seed_label"] == "math-tranche-configurations"
    assert draw_cli.STRING_TASK["bbh"]["seed_label"] == draw_cli.BBH_SEED_LABEL == "bbh-tranche-configurations"


def test_forced_math_decoding_differs_only_in_the_stop_string():
    base = yaml.safe_load((ROOT / "config" / "decoding_forced.yaml").read_text(encoding="utf-8"))
    m = yaml.safe_load((ROOT / "config" / "decoding_forced_math.yaml").read_text(encoding="utf-8"))
    diff = {k for k in set(base) | set(m) if base.get(k) != m.get(k)}
    assert diff <= {"status", "stop", "include_stop_str_in_output"} and m["stop"] == []
    assert m["prefill"] == {"forced_direct": "\\boxed{"} and m["max_new_tokens"] == {"forced_direct": 16}
    assert m["status"] == "confirmed" and gen.load_decoding(ROOT / "config" / "decoding_forced_math.yaml").confirmed
    assert list(sfd_cli.box_closed(["\\boxed{\\frac{14}", "\\boxed{\\frac{14}{3}}"], "math")) == [False, True]
    assert list(sfd_cli.box_closed(["\\boxed{12", "\\boxed{12}"])) == [False, True]


def metric_rows():
    """Row metrics of four models on two configurations: m_long answers at length in the direct arm, m_think writes
    think tags, c_cut is truncated in the chain-of-thought arm."""
    rows = []
    for m in ("m_ok", "m_ok2", "m_long", "m_think"):
        for c in ("c_ok", "c_cut"):
            for arm in ("direct_answer", "zero_shot_cot"):
                for i in range(10):
                    rows.append({"model_key": m, "arm": arm, "config_id": c, "item_id": f"train/algebra/{i}",
                                 "n_output_tokens": 400 if (m == "m_long" and arm == "direct_answer") else 20,
                                 "registered": True, "lenient": True, "think_tag": m == "m_think", "empty": False,
                                 "truncated": c == "c_cut" and arm == "zero_shot_cot" and i < 2})
    return pd.DataFrame(rows)


def test_direct_answer_rule_is_a_diagnostic_for_math():
    rules, _ = screening.load_rules()
    assert math_d2.DIAGNOSTIC_MODEL_RULES == ("direct_answer_meaningful",)
    assert not hasattr(bbh_d2, "DIAGNOSTIC_MODEL_RULES")
    rows = metric_rows()
    applied = screening.apply_rules(rows, rules, has_registered=True)
    assert "model_diagnostics" not in applied                                  # GSM8K and BBH: unchanged
    assert sorted({h["model_key"] for h in applied["excluded_models"]}) == ["m_long", "m_think"]
    diag = screening.apply_rules(rows, rules, has_registered=True, diagnostic_model_rules=math_d2.DIAGNOSTIC_MODEL_RULES)
    assert diag["retained_models"] == ["m_long", "m_ok", "m_ok2"]
    assert [h["model_key"] for h in diag["excluded_models"]] == ["m_think"]    # the think-tag rule still applies
    assert [(h["model_key"], h["value"]) for h in diag["model_diagnostics"]["flagged"]] == [("m_long", 100.0)]
    assert [c["config_id"] for c in diag["excluded_configurations"]] == ["c_cut"]          # truncation 20% > 2%
    assert diag["excluded_configurations"][0]["violations"][0]["pooled_over_models"] == ["m_long", "m_ok", "m_ok2"]
    assert {k: v for k, v in diag.items() if k not in ("model_diagnostics", "excluded_models", "retained_models",
                                                         "excluded_configurations")} == \
        {k: v for k, v in applied.items() if k not in ("excluded_models", "retained_models", "excluded_configurations")}
    with pytest.raises(screening.ScreeningError, match="unknown model rules"):
        screening.apply_rules(rows, rules, has_registered=True, diagnostic_model_rules=("max_truncation_pct",))


def test_math_truncation_rule_is_relative_to_the_arm_median():
    rules, _ = screening.load_rules()
    assert math_d2.RELATIVE_TRUNCATION is True and not hasattr(bbh_d2, "RELATIVE_TRUNCATION")
    rows = []
    cut = {"c_a": 3, "c_b": 3, "c_c": 8}        # chain-of-thought outputs cut per model of 50: 6%, 6% and 16%
    for m in ("m_ok", "m_ok2"):
        for c in cut:
            for arm in ("direct_answer", "zero_shot_cot"):
                for i in range(50):
                    rows.append({"model_key": m, "arm": arm, "config_id": c, "item_id": f"train/algebra/{i}",
                                 "n_output_tokens": 20, "registered": True, "lenient": True, "think_tag": False,
                                 "empty": False, "truncated": arm == "zero_shot_cot" and i < cut[c]})
    rows = pd.DataFrame(rows)
    absolute = screening.apply_rules(rows, rules, has_registered=True)
    assert [c["config_id"] for c in absolute["excluded_configurations"]] == ["c_a", "c_b", "c_c"]   # all above 2%
    relative = screening.apply_rules(rows, rules, has_registered=True, relative_truncation=True)
    assert [c["config_id"] for c in relative["excluded_configurations"]] == ["c_c"]    # 16% is 10 points above 6%
    (v,) = relative["excluded_configurations"][0]["violations"]
    assert (v["rule"], v["metric"], v["arm"]) == ("max_truncation_pct", "truncated_pct_above_arm_median", "zero_shot_cot")
    assert (v["value"], v["threshold"], v["truncated_pct"], v["arm_median_truncated_pct"]) == (10.0, 2.0, 16.0, 6.0)
    # the think-tag rule stays absolute
    rows.loc[(rows["config_id"] == "c_a") & (rows["item_id"] == "train/algebra/0"), "think_tag"] = True
    again = screening.apply_rules(rows, rules, has_registered=True, relative_truncation=True)
    assert [c["config_id"] for c in again["excluded_configurations"]] == ["c_a", "c_c"]
    assert [v["rule"] for v in again["excluded_configurations"][0]["violations"]] == ["max_think_tag_pct"] * 2


def test_math_promptsuite_config_differs_from_bbh_in_three_fields():
    bbh_cfg = yaml.safe_load((ROOT / "config" / "promptsuite_bbh.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config" / "promptsuite_math.yaml").read_text(encoding="utf-8"))
    assert (cfg["name"], cfg["seed"], cfg["instruction"]["text"]) == ("math_bank_v1", 20261103,
                                                                       "Solve the following math problem.")
    assert cfg["answer_format"]["text"] == mt.ANSWER_FORMAT
    assert (cfg["llm"]["record"], cfg["llm"]["review"]) == ("manifests/math_paraphrases.json",
                                                            "manifests/math_paraphrase_review.json")
    same = {k for k in cfg if cfg[k] == bbh_cfg[k]}
    assert set(cfg) - same == {"name", "seed", "instruction", "answer_format", "llm"}
    assert {k for k in cfg["instruction"] if cfg["instruction"][k] != bbh_cfg["instruction"][k]} == {"text"}
    assert {k for k in cfg["llm"] if cfg["llm"][k] != bbh_cfg["llm"][k]} == {"record", "review"}
    req = ROOT / "data" / "bank_candidates" / "paraphrase_math" / "request.json"
    if req.is_file():                                                          # the rendered request, when present
        r = json.loads(req.read_text(encoding="utf-8"))
        assert (r["text"], r["field"], r["n_augments"], r["promptsuite_version"]) == (
            "Solve the following math problem.", "instruction", 7, "3.0.7")


# ---------------------------------------------------------------- the whole chain on fabricated data

# ---------------------------------------------------------------- the pinned item tables

needs_tables = pytest.mark.skipif(
    not all((ROOT / "data" / "processed" / "math" / f"items_{s}.parquet").is_file()
            for s in ("screening", "calibration", "target")),
    reason="MATH item tables absent (scripts/fetch_math.py)")


@needs_tables
def test_real_item_sets():
    for name, n in (("screening", 200), ("calibration", 398), ("target", 398)):
        df, info = math_d2.load_items(name)
        assert len(df) == n and info["matches_manifest"] is True
    reg = math_d2.load_registries()
    assert all(k in reg for k in math_d2.panel()) and len(math_d2.panel()) == 29


@needs_tables
@pytest.mark.skipif(not (ROOT / "manifests" / "gsm8k_bank.json").is_file(), reason="GSM8K bank absent")
def test_real_screening_requests_render_math_items(tmp_path):
    """Request shapes on the real screening items, with the frozen GSM8K bank as a stand-in for the MATH bank."""
    keys = ["qwen3-8b", "qwen3-14b"]                     # one from each registry
    assert build_cli.main(["--task", "math", "--bank", str(ROOT / "manifests" / "gsm8k_bank.json"), "--models", *keys,
                           "--screening", "--out-dir", str(tmp_path)]) == 0
    rec = json.loads((tmp_path / "screening" / "build_record.json").read_text(encoding="utf-8"))
    assert rec["task"] == "math" and rec["items"]["n_items"] == 200 and rec["split"] == "train"
    for k in keys:
        df = pd.read_parquet(tmp_path / "screening" / f"{k}.parquet")
        assert len(df) == 56 * 2 * 200 == 22400 and df["item_id"].str.startswith("train/").all()
        msgs = json.loads(df.loc[0, "messages_json"])
        assert "\\boxed{<number>}" in msgs[-1]["content"] and "{question}" not in msgs[-1]["content"]


def test_collection_registry_and_decoding():
    """config/models_math.yaml is the panel's entries with max_model_len 8,192 (4,096 for the native-4k checkpoints), and
    config/decoding_math_4k.yaml equals config/decoding.yaml but for the chain-of-thought cap."""
    base = math_d2.load_registries()
    reg = math_d2.collection_registry()
    assert len(reg) == 29 and set(math_d2.NATIVE_4K) <= set(reg)
    for k, e in reg.items():
        assert e == {**base[k], "max_model_len": 4096 if k in math_d2.NATIVE_4K else 8192}
    assert yaml.safe_load(math_d2.COLLECTION_REGISTRY.read_text(encoding="utf-8"))["dataset2"]["models"] == reg
    d, d4 = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in (math_d2.DECODING, math_d2.DECODING_4K))
    assert d4["max_new_tokens"] == {"direct_answer": 1024, "zero_shot_cot": 2900} and d4["status"] == "confirmed"
    assert {k: v for k, v in d4.items() if k != "max_new_tokens"} == {k: v for k, v in d.items() if k != "max_new_tokens"}
    assert math_d2.decoding_for("llama-2-7b-chat-hf") == math_d2.DECODING_4K
    assert math_d2.decoding_for("qwen3-8b") == math_d2.DECODING
    assert math_d2.decoding_for("llama-2-7b-chat-hf", forced=True) == math_d2.DECODING_FORCED


@needs_tables
@pytest.mark.skipif(not (ROOT / "data" / "tokenizers").is_dir(), reason="local tokenizers absent")
def test_longest_prompt_fits_the_native_4k_cap():
    """The longest calibration or target problem, the longest frame of the screening requests (46 tokens) and 100
    tokens of template markup leave room for the 2,900-token cap under 4,096."""
    from tokenizers import Tokenizer
    qs = pd.concat([math_d2.load_items(s)[0] for s in ("calibration", "target")])["question"]
    for k in math_d2.NATIVE_4K:
        tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizers" / k / "tokenizer.json"))
        longest = max(len(tok.encode(q, add_special_tokens=False).ids) for q in qs)
        assert longest + 46 + 100 + 2900 <= 4096, (k, longest)
