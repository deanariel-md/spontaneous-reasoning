"""Tests for BIG-Bench Hard in the Dataset 2 pipeline (src/bbh_d2.py and the --task bbh paths of
scripts/build_requests.py, score_d2.py, screen_d2.py, freeze_bank_d2.py, draw_tranche_d2.py, freeze_sampler_d2.py,
run_dataset2.py and precision_matched.py).

The chain test runs on fabricated item tables, a fabricated items manifest, a fabricated candidate bank and four
fabricated models in two registries, with the real runner and an echo backend whose outputs are fabricated. Tests
on the pinned BIG-Bench Hard files, the real candidate bank and the real registries skip when they are absent.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src import bbh, bbh_d2, d2_io, generation as gen, screening

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
ARMS = ("direct_answer", "zero_shot_cot")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_bbh_cli", ROOT / "scripts" / f"{name}.py")
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


# ---------------------------------------------------------------- fabricated inputs

TASK_GOLD = {"snarks": ("letter", ["(A)", "(B)", "(B)"]),
             "boolean_expressions": ("words:True/False", ["True", "False", "True"]),
             "object_counting": ("integer", ["12", "-3", "1000"])}
SET_INDICES = {"screening": (0, 2), "calibration": (2, 4), "target": (4, 7)}


def item_rows(item_set):
    lo, hi = SET_INDICES[item_set]
    rows = []
    for task, (label, golds) in TASK_GOLD.items():
        for i in range(lo, hi):
            rows.append({"item_id": f"{task}/{i}", "question": f"Describe {task}.\n\nItem {task} number {i} asks this.",
                         "gold_answer": golds[i % len(golds)], "task": task, "answer_type": label})
    return pd.DataFrame(rows, columns=list(bbh.TABLE_COLUMNS))


def write_items(tmp):
    man = {"revision": bbh.COMMIT, "tasks": list(bbh.TASKS), "tables": {}, "sets": {},
           "per_task": {t: {"answer_type": label} for t, (label, _) in TASK_GOLD.items()}}
    for name in SET_INDICES:
        df = item_rows(name)
        raw = bbh.table_bytes(df)
        p = tmp / f"items_{name}.parquet"
        p.write_bytes(raw)
        man["tables"][name] = {"path": str(p), "n_rows": len(df), "sha256": hashlib.sha256(raw).hexdigest()}
        man["sets"][name] = {"n_items": len(df), "canonical_sha256": bbh.canonical_sha256(df)}
    path = tmp / "bbh_items.json"
    path.write_text(json.dumps(man), encoding="utf-8")
    return path, man


def make_candidate():
    def cfg(instr):
        def arm(clause):
            return {"messages": [{"role": "user", "content": f"{instr}\n\nQuestion: {{question}}\n{clause} End your reply "
                                                             "with a final line that contains \\boxed{<answer>}."}]}
        return {"dimensions": {"instruction": instr},
                "arms": {"direct_answer": arm("Give only the final answer, without any working."),
                         "zero_shot_cot": arm("Let's think step by step.")}}
    ids = ["c1", "c2", "c3"]
    return {"generator": "handwritten-bbh-test", "generator_version": "0", "seed": 20261031,
            "generation_record": {"config_status": "confirmed", "sampling_law": "uniform"},
            "sampling_probabilities": {c: 1 / 3 for c in ids},
            "configurations": {"c1": cfg("Answer the following question."), "c2": cfg("Respond to the query."),
                               "c3": cfg("NOBOX Reply to the question.")}}


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


WRONG = {"(A)": "(B)", "(B)": "(A)", "True": "False", "False": "True"}


class BBHEcho(gen.EchoBackend):
    """Echo backend with fabricated BIG-Bench Hard answers: a final \\boxed{} line (none for configuration c3, whose
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
            ans = gold if ok else WRONG.get(gold, str(int(gold) + 1) if gold.lstrip("-").isdigit() else gold)
            direct = "Give only the final answer" in user
            body = "word " * 200 if (self.spec.model_key == "m3" and direct) else ("" if direct else "Some steps.\n")
            text = body + ("The answer follows." if "NOBOX" in user else f"\\boxed{{{ans}}}")
            out.append(gen.GenResult(text, r.n_prompt_tokens, len(text.split()), "stop", r.prompt_echo))
        return out


class Env:
    def __init__(self, tmp, monkeypatch):
        self.tmp = tmp
        self.items_manifest, self.man = write_items(tmp)
        monkeypatch.setattr(bbh_d2, "ITEMS_MANIFEST", self.items_manifest)
        self.cand_path = tmp / "candidate.json"
        self.cand_path.write_text(json.dumps(make_candidate()), encoding="utf-8")
        self.reg1, self.reg2 = tmp / "models.yaml", tmp / "models_extension.yaml"
        self.reg1.write_text(yaml.safe_dump({"dataset2": {"models": {
            "_template": {"hf_id": "PLACEHOLDER"}, "m1": model_entry(family="Alpha", hf_id="org/m1"),
            "m2": model_entry(family="Beta", hf_id="org/m2")}}}))
        self.reg2.write_text(yaml.safe_dump({"dataset2": {"models": {
            "m3": model_entry(family="Beta", hf_id="org/m3"), "m4": model_entry(family="Alpha", hf_id="org/m4")}}}))
        monkeypatch.setattr(bbh_d2, "REGISTRIES", (self.reg1, self.reg2))
        panel = {"panel": [{"repo": f"org/m{i}", "tier": t} for i, t in ((1, "low"), (2, "middle"), (3, "high"),
                                                                         (4, "high"))]}
        self.panel_path = tmp / "panel.json"
        self.panel_path.write_text(json.dumps(panel), encoding="utf-8")
        monkeypatch.setattr(bbh_d2, "PANEL_EXTENSION", self.panel_path)
        self.decoding = tmp / "decoding.yaml"
        self.decoding.write_text(yaml.safe_dump(decoding_dict()))
        self.req = tmp / "requests"
        self.questions = {}
        for name in SET_INDICES:
            df = item_rows(name)
            self.questions.update(dict(zip(df["question"], df["gold_answer"])))

    def build(self, *extra):
        return build_cli.main(["--task", "bbh", "--out-dir", str(self.req), "--decoding-yaml", str(self.decoding),
                               *map(str, extra)])

    def generate(self, key, sub):
        reg = Path(bbh_d2.registry_for(key))              # the runner's MODELS_YAML for this model
        reg = reg if reg.is_absolute() else ROOT / reg
        spec = gen.parse_model_spec(key, gen.load_registry(reg))
        dec = gen.load_decoding(self.decoding)
        out = self.tmp / "gen" / (sub or "main") / key
        req = self.req / sub / f"{key}.parquet" if sub else self.req / f"{key}.parquet"
        log = gen.run_generation(req, key, out, backend="echo", backend_obj=BBHEcho(spec, dec, self.questions),
                                 models_path=reg, decoding_path=self.decoding, verbose=False, shard_size=1000)
        assert log["status"] == "complete"
        return out

    def score(self, dirs, out, *extra):
        return score_cli.main(["--task", "bbh", "--work-dir", *map(str, dirs), "--requests-dir", str(self.req),
                               "--out", str(out), "--allow-test-backend", *extra])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "gpu_info", lambda timeout=10.0: {"count": 0, "devices": [], "driver_version": None,
                                                              "cuda_version_driver": None, "nvidia_smi": False})
    return Env(tmp_path, monkeypatch)


# ---------------------------------------------------------------- items, scorer, registries

def test_load_items_checks_the_manifest(env, tmp_path):
    for name, n in (("screening", 6), ("calibration", 6), ("target", 9)):
        df, info = bbh_d2.load_items(name)
        assert len(df) == n and list(df.columns) == list(bbh_d2.ITEM_COLUMNS)
        assert info["item_set"] == name and info["split"] == ("test" if name == "target" else "train")
        assert info["canonical_sha256"] == env.man["sets"][name]["canonical_sha256"] and info["matches_manifest"]
    p = Path(env.man["tables"]["target"]["path"])
    df = pd.read_parquet(p)
    df.loc[0, "gold_answer"] = "(B)" if df.loc[0, "gold_answer"] == "(A)" else "(A)"
    df.to_parquet(p, index=False)
    with pytest.raises(bbh_d2.BBHPipelineError, match="differs"):
        bbh_d2.load_items("target")
    with pytest.raises(bbh_d2.BBHPipelineError, match="item set"):
        bbh_d2.load_items("train")


def test_custom_items_are_labelled_dry_run(env, tmp_path):
    p = tmp_path / "custom.parquet"
    item_rows("target").to_parquet(p, index=False)
    df, info = bbh_d2.load_items("target", p)
    assert info["matches_manifest"] is None and "dry run" in info["source"]
    bad = item_rows("target").assign(gold_answer="B")
    bad.to_parquet(p, index=False)
    with pytest.raises(bbh_d2.BBHPipelineError, match="canonical"):
        bbh_d2.load_items("target", p)


def test_load_scorer(tmp_path):
    rec = bbh_d2.load_scorer()
    assert rec["status"] == "registered" and rec["scorer_version"] == bbh.SCORER_VERSION
    p = tmp_path / "scorer.json"
    p.write_text(json.dumps({**bbh.scorer_record(), "scorer_version": "bbh-boxed/0.9"}), encoding="utf-8")
    with pytest.raises(bbh_d2.BBHPipelineError, match="differs"):
        bbh_d2.load_scorer(p)


def test_registries_union_and_tiers(env, tmp_path):
    reg = bbh_d2.load_registries()
    assert sorted(reg) == ["m1", "m2", "m3", "m4"]
    assert bbh_d2.extension_tiers(reg, ["m1", "m2", "m4"]) == {"m1": "low", "m2": "middle", "m4": "high"}
    assert bbh_d2.extension_tiers({**reg, "m5": model_entry(hf_id="org/none")}, ["m1", "m5"]) is None
    assert Path(bbh_d2.registry_for("m3")).name == "models_extension.yaml"
    assert Path(bbh_d2.registry_for("m1")).name == "models.yaml"
    with pytest.raises(bbh_d2.BBHPipelineError, match="none of the registries"):
        bbh_d2.registry_for("zz")
    dup = tmp_path / "dup.yaml"
    dup.write_text(yaml.safe_dump({"dataset2": {"models": {"m1": model_entry()}}}))
    with pytest.raises(bbh_d2.BBHPipelineError, match="two registries"):
        bbh_d2.load_registries([env.reg1, dup])


def test_render_keeps_string_item_ids():
    bank = {"generator": "g", "generator_version": "0", "seed": 0, "sampling_probabilities": {"c": 1.0},
            "configurations": {"c": {"dimensions": {}, "arms": {a: {"messages": [{"role": "user", "content": "Q {question}"}]}
                                                                 for a in ARMS}}}}
    items = pd.DataFrame({"item_id": ["snarks/3", "web_of_lies/10"], "question": ["a", "b"]})
    df = d2_io.render_requests(bank, ["c"], "m", items, split="train", string_item_ids=True)
    assert df["item_id"].tolist() == ["snarks/3", "web_of_lies/10"] * 2
    assert df["request_id"].tolist()[0] == "train|c|direct_answer|snarks/3"
    with pytest.raises(d2_io.BankError, match="string item ids"):
        d2_io.render_requests(bank, ["c"], "m", items.assign(item_id=["a|b", "c"]), string_item_ids=True)


# ---------------------------------------------------------------- the whole chain on fabricated data

def test_analysis_runs_on_string_item_ids(tmp_path):
    """run_dataset2.run on simulated outputs with BIG-Bench Hard item ids: predictions saved before the validation
    units load, results/<label>.json and the tables in tables_<label>, nothing in the GSM8K tables folder."""
    from src.simulate import GSM8KSim
    from src import sampling
    sim = GSM8KSim(np.random.default_rng(3), M=8, C=40, n_items=40)
    sampler = sampling.registered_split_bank(sim.config_ids, np.full(sim.C, 1 / sim.C), 8, 8, "bbh")
    names = [f"{bbh.TASKS[i % len(bbh.TASKS)]}/{i}" for i in range(sim.N)]
    pred = str(tmp_path / "predictions_bbh.json")
    calls = []

    def load(ids):
        calls.append(os.path.exists(pred))
        df = sim.scored(ids)
        df["item_id"] = df["item_id"].map(lambda i: names[int(i)]).astype(str)
        return df

    res = tmp_path / "res"
    out = rd2.run(load, sampler, sim.panel, sim.families, sim.tiers, names, str(res), pred, "bbh", n_boot=199, n_ctrl=20,
                  tables_dir=str(res / "tables_bbh"))
    assert calls[0] is False and all(calls[1:])
    assert (res / "bbh.json").exists() and (tmp_path / "predictions_bbh_lenient.json").exists()
    assert (res / "tables_bbh" / "steps12_registered.csv").exists() and not (res / "tables").exists()
    assert out["label"] == "bbh" and out["step3"]["frozen_sha256"] == json.load(open(pred))["sha256"]


def test_registered_bbh_analysis_refuses_gsm8k_and_echo_tables(tmp_path):
    p = tmp_path / "scored.parquet"
    t = pa.table({"model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": ["snarks/0"],
                  "strict_correct": [1], "lenient_correct": [1], "compliant": [True]})
    inputs = [{"model_key": "m", "test_backend": True, "request_kind": "main", "split": "test",
               "incomplete_allowed": False, "decoding_confirmed": True}]
    pq.write_table(t.replace_schema_metadata({"d2_kind": "main", "d2_split": "test",
                                             "d2_scorer_status": "registered", "d2_inputs": json.dumps(inputs)}), p)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_dataset2.py"), "--task", "bbh", "--scored", str(p)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "d2_task" in r.stderr and "test backend" in r.stderr
    md = {"d2_kind": "main", "d2_split": "test", "d2_scorer_status": "registered",
          "d2_inputs": json.dumps(inputs), "d2_task": "bbh"}
    pq.write_table(t.replace_schema_metadata(md), p)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_dataset2.py"), "--scored", str(p)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--task bbh" in r.stderr


def test_bbh_tranche_config_is_fixed():
    t = yaml.safe_load((ROOT / "config" / "tranche_bbh.yaml").read_text(encoding="utf-8"))
    assert t == {"status": "confirmed", "configurations": 8, "seed": 20261031, "calibration_items": 1350}
    assert draw_cli.BBH_SEED_LABEL == "bbh-tranche-configurations"


# ---------------------------------------------------------------- the pinned files and the real inputs

needs_tables = pytest.mark.skipif(
    not all((ROOT / "data" / "processed" / "bbh" / f"items_{s}.parquet").is_file()
            for s in ("screening", "calibration", "target")) or not bbh_d2.CANDIDATE_BANK.is_file(),
    reason="BIG-Bench Hard item tables or the candidate bank absent")


@needs_tables
def test_real_item_sets_and_registries():
    for name, n in (("screening", 200), ("calibration", 1350), ("target", 1350)):
        df, info = bbh_d2.load_items(name)
        assert len(df) == n and info["matches_manifest"] is True
    reg = bbh_d2.load_registries()
    assert len(reg) == 30
    tiers = bbh_d2.extension_tiers(reg, list(reg))
    assert tiers is not None and set(tiers.values()) == {"low", "middle", "high"}


@needs_tables
def test_real_screening_requests_have_64_configurations_2_arms_200_items(tmp_path):
    keys = ["qwen3-8b", "qwen3-14b"]                     # one from each registry
    assert build_cli.main(["--task", "bbh", "--bank", str(bbh_d2.CANDIDATE_BANK), "--models", *keys,
                           "--screening", "--out-dir", str(tmp_path)]) == 0
    rec = json.loads((tmp_path / "screening" / "build_record.json").read_text(encoding="utf-8"))
    assert rec["n_draws"] == 64 and rec["items"]["n_items"] == 200 and rec["split"] == "train"
    for k in keys:
        df = pd.read_parquet(tmp_path / "screening" / f"{k}.parquet")
        assert len(df) == 64 * 2 * 200 == 25600
        assert df["config_id"].nunique() == 64 and df["item_id"].nunique() == 200
        assert df.groupby("arm").size().to_dict() == {"direct_answer": 12800, "zero_shot_cot": 12800}
        assert rec["models"][k]["n_requests"] == 25600
        assert gen.calibration_build(tmp_path / "screening" / f"{k}.parquet", k)["kind"] == "screening"
        msgs = json.loads(df.loc[0, "messages_json"])
        assert "\\boxed{<answer>}" in msgs[-1]["content"] and "{question}" not in msgs[-1]["content"]
