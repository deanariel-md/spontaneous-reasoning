"""Tests for src/d2_io.py, scripts/build_requests.py and scripts/score_d2.py.

Fabricated bank, items and models in tmp_path; the runner is the real one with the EchoBackend. Nothing
needs a GPU, vLLM, torch or the network. Data-dependent tests skip when data/raw/gsm8k is absent.
"""
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from src import d2_io, dataset2 as d2, generation as gen, gsm8k, sampling, scoring

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
ARMS = ("direct_answer", "zero_shot_cot")
N_ITEMS = 5


# ---------------------------------------------------------------- fabricated inputs

def make_bank():
    def cfg(dim, direct, cot, system=None):
        def pre():
            return [{"role": "system", "content": system}] if system else []
        return {"dimensions": {"wording": dim},
                "arms": {"direct_answer": {"messages": pre() + [{"role": "user", "content": direct}]},
                         "zero_shot_cot": {"messages": pre() + [{"role": "user", "content": cot}]}}}
    return {
        "generator": "handwritten-test", "generator_version": "0", "seed": 0,
        "sampling_probabilities": {"c1": 0.25, "c2": 0.75},
        "configurations": {
            "c1": cfg("plain",
                      'Q: {question}\nFinal number only, "Answer: N". JSON like {"a": 1}, \\boxed{}, {{x}}, {answer}.',
                      'Q: {question}\nThink step by step, then end with "Answer: N". JSON like {"a": 1}, {{x}}, {answer}.'),
            "c2": cfg("formal", "Problem: {question}\nReply with the final number, as 'Answer: N'.",
                      "Problem: {question}\nReason step by step; the last line must be 'Answer: N'.",
                      system="You are a careful solver.")}}


def make_items(n=N_ITEMS):
    q = [f"Compute {i}+{i} (use {{braces}} freely), café." for i in range(n)]
    q[3] = "Explain why {question} is a placeholder, then compute 3+3."
    return pd.DataFrame({"item_id": pd.Series(range(n), dtype="int64"), "question": q,
                         "gold_answer": [str(2 * i) for i in range(n)]})


def model_entry(**over):
    e = {"family": "test", "hf_id": "org/model", "revision": SHA, "dtype": "bfloat16", "quantization": "none",
         "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": 1,
         "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
         "thinking_off": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False},
                          "system_prompt_token": None, "rationale": None}, "kernels": "batch_invariant"}
    e.update(over)
    return e


def decoding_dict(**over):
    d = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
         "max_new_tokens": {"direct_answer": 400, "zero_shot_cot": 400}, "stop": [],
         "include_stop_str_in_output": False, "skip_special_tokens": True,
         "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                    "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}
    d.update(over)
    return d


def admissibility_log(configs, models=("m1", "m2", "zz_model"), excluded_models=(), excluded_configs=()):
    """A fabricated exclusion log in the shape scripts/screen_d2.py writes (the fields the guards read)."""
    return {"kind": "admissibility_exclusion_log", "models_screened": [*models, *excluded_models],
            "excluded_models": [{"model_key": m, "rule": "direct_answer_meaningful"} for m in excluded_models],
            "retained_models": list(models), "configuration_rules_evaluated": True,
            "configurations_screened": [*configs, *excluded_configs],
            "excluded_configurations": [{"config_id": c, "violations": []} for c in excluded_configs],
            "retained_configuration_ids": list(configs)}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_cli = load_script("build_requests")
score_cli = load_script("score_d2")


class Env:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.bank_path = tmp_path / "bank.json"
        self.bank = make_bank()
        self.bank_path.write_text(json.dumps(self.bank, ensure_ascii=False), encoding="utf-8")
        self.items = make_items()
        self.items_path = tmp_path / "items.parquet"
        self.items.to_parquet(self.items_path, index=False)
        self.models = tmp_path / "models.yaml"
        self.models.write_text(yaml.safe_dump({"dataset2": {"models": {
            "_template": {"hf_id": "PLACEHOLDER"}, "m1": model_entry(), "m2": model_entry(family="other")}}}))
        self.decoding = tmp_path / "decoding.yaml"
        self.decoding.write_text(yaml.safe_dump(decoding_dict()))
        self.req_dir = tmp_path / "requests"
        self.sampler_path = tmp_path / "sampler.json"
        self.write_sampler()
        self.adm_path = tmp_path / "admissibility.json"
        self.write_admissibility()

    def write_admissibility(self, **kw):
        log = admissibility_log(list(self.bank["configurations"]), **kw)
        self.adm_path.write_text(json.dumps(log), encoding="utf-8")
        return log

    def write_sampler(self, label="t6"):
        _, sha = d2_io.load_bank(self.bank_path)
        ids = list(self.bank["configurations"])
        p = [self.bank["sampling_probabilities"][i] for i in ids]
        s = sampling.registered_split_bank(ids, p, 6, 6, label)
        s["bank_file_sha256"] = sha
        self.sampler_path.write_text(json.dumps(s), encoding="utf-8")
        self.sampler = s
        return s

    def build_args(self, *models, extra=(), sampler=True):
        a = ["--bank", str(self.bank_path), "--models", *models, "--out-dir", str(self.req_dir),
             "--items-parquet", str(self.items_path), "--allow-unregistered", "--models-yaml", str(self.models),
             "--decoding-yaml", str(self.decoding), "--admissibility", str(self.adm_path)]
        if sampler:
            a += ["--sampler", str(self.sampler_path)]
        return a + list(extra)

    def build(self, *models, extra=(), sampler=True):
        return build_cli.main(self.build_args(*models, extra=extra, sampler=sampler))

    def record(self, tranche=False, sub=None):
        sub = sub or ("tranche" if tranche else "")
        p = self.req_dir / sub / "build_record.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def requests(self, key, tranche=False, sub=None):
        return self.req_dir / (sub or ("tranche" if tranche else "")) / f"{key}.parquet"

    def use_train_items(self, n=12):
        """Fabricated stand-in for the train split (the dry runs read it through --items-parquet)."""
        self.items = make_items(n)
        self.items.to_parquet(self.items_path, index=False)
        self.cal_path = self.tmp / "calibration_items.json"
        return self.items

    def build_train(self, *models, n=5, mode="--screening", extra=()):
        """Build train-split requests (screening by default); the sample goes to the tmp manifest, never the repo's."""
        if not hasattr(self, "cal_path"):
            self.use_train_items()
        return self.build(*models, sampler=False, extra=[
            "--split", "train", "--calibration-items", str(n), "--calibration-manifest", str(self.cal_path), mode, *extra])

    def generate(self, key, out_dir, tranche=False, backend="echo", sub=None, **kw):
        spec = gen.parse_model_spec(key, gen.load_registry(self.models))
        dec = gen.load_decoding(self.decoding)
        be = GoldEcho(spec, dec, dict(zip(self.items["question"], self.items["gold_answer"]))) if backend == "gold" \
            else gen.EchoBackend(spec, dec)
        return gen.run_generation(self.requests(key, tranche, sub), key, out_dir, backend="echo", backend_obj=be,
                                  models_path=self.models, decoding_path=self.decoding, verbose=False,
                                  shard_size=1000, **kw)

    def score_args(self, *work_dirs, out=None, extra=("--candidate-spec", "answer_colon", "--allow-test-backend")):
        return ["--work-dir", *map(str, work_dirs), "--out", str(out or self.tmp / "scored.parquet"),
                "--requests-dir", str(self.req_dir), "--items-parquet", str(self.items_path), *extra]

    def score(self, *work_dirs, out=None, extra=("--candidate-spec", "answer_colon", "--allow-test-backend")):
        return score_cli.main(self.score_args(*work_dirs, out=out, extra=extra))


class GoldEcho(gen.EchoBackend):
    """EchoBackend whose output ends in 'Answer: N'; N is the gold answer for half of the prompts."""

    def __init__(self, spec, decoding, gold_by_question):
        super().__init__(spec, decoding)
        self.gold = gold_by_question

    def generate(self, conversations, params):
        res = super().generate(conversations, params)
        out = []
        for conv, r in zip(conversations, res):
            user = conv[-1]["content"]
            q = max((q for q in self.gold if q in user), key=len)
            g = int(self.gold[q])
            ok = hashlib.sha256(r.prompt_echo.encode("utf-8")).digest()[0] % 2 == 0
            text = f"Let me work it out.\nAnswer: {g if ok else g + 1}"
            out.append(gen.GenResult(text, r.n_prompt_tokens, len(text.split()), "stop", r.prompt_echo))
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "gpu_info", lambda timeout=10.0: {"count": 0, "devices": [], "driver_version": None,
                                                              "cuda_version_driver": None, "nvidia_smi": False})
    return Env(tmp_path)


# ---------------------------------------------------------------- bank validation

def test_load_bank_returns_the_parsed_bank_and_the_file_hash(env):
    bank, sha = d2_io.load_bank(env.bank_path)
    assert bank == env.bank
    assert sha == hashlib.sha256(env.bank_path.read_bytes()).hexdigest()
    assert d2_io.validate_bank(bank) == []


def _drop_arm(b):
    del b["configurations"]["c1"]["arms"]["zero_shot_cot"]


def _extra_arm(b):
    b["configurations"]["c1"]["arms"]["few_shot"] = copy.deepcopy(b["configurations"]["c1"]["arms"]["direct_answer"])


def _no_token(b):
    m = b["configurations"]["c1"]["arms"]["direct_answer"]["messages"]
    m[0]["content"] = m[0]["content"].replace("{question}", "QUESTION")


def _token_twice_in_one_message(b):
    m = b["configurations"]["c1"]["arms"]["direct_answer"]["messages"]
    m[0]["content"] += " again {question}"


def _token_twice_across_messages(b):
    m = b["configurations"]["c2"]["arms"]["zero_shot_cot"]["messages"]
    m[0]["content"] += " {question}"                      # the system message gains a second occurrence


def _token_in_system_only(b):
    for a in ARMS:                                        # moved to the system message: still exactly once
        m = b["configurations"]["c2"]["arms"][a]["messages"]
        m[1]["content"] = m[1]["content"].replace("{question}", "QUESTION")
        m[0]["content"] += " {question}"


def _negative_p(b):
    b["sampling_probabilities"] = {"c1": -0.25, "c2": 1.25}


def _sum_off(b):
    b["sampling_probabilities"] = {"c1": 0.25, "c2": 0.75 + 3e-9}


def _sum_within(b):
    b["sampling_probabilities"] = {"c1": 0.25, "c2": 0.75 + 5e-10}


def _missing_p(b):
    del b["sampling_probabilities"]["c2"]


def _extra_p(b):
    b["sampling_probabilities"]["c3"] = 0.0


def _list_p(b):
    b["sampling_probabilities"] = [0.25, 0.75]


def _last_role(b):
    b["configurations"]["c1"]["arms"]["direct_answer"]["messages"].append({"role": "assistant", "content": "x"})


def _extra_message_key(b):
    b["configurations"]["c1"]["arms"]["direct_answer"]["messages"][0]["name"] = "bob"


def _bad_role(b):
    b["configurations"]["c1"]["arms"]["direct_answer"]["messages"][0]["role"] = "tool"


def _no_generator(b):
    del b["generator_version"]


def _empty_seed(b):
    b["seed"] = ""


def _no_configs(b):
    b["configurations"] = {}


def _bad_id(b):
    b["configurations"]["a|b"] = b["configurations"]["c1"]
    b["sampling_probabilities"]["a|b"] = 0.0


def _no_dimensions(b):
    del b["configurations"]["c1"]["dimensions"]


@pytest.mark.parametrize("mutate, match", [
    (_drop_arm, "arm 'zero_shot_cot' is missing"),
    (_extra_arm, "unexpected arm 'few_shot'"),
    (_no_token, r"\{question\} occurs 0 times"),
    (_token_twice_in_one_message, r"\{question\} occurs 2 times"),
    (_token_twice_across_messages, r"\{question\} occurs 2 times"),
    (_negative_p, "negative"),
    (_sum_off, "not 1 within"),
    (_missing_p, "without a sampling probability"),
    (_extra_p, "without a configuration"),
    (_list_p, "must be a mapping"),
    (_last_role, "last message must have role user"),
    (_extra_message_key, "exactly the keys role and content"),
    (_bad_role, "role 'tool'"),
    (_no_generator, "generator_version"),
    (_empty_seed, "seed"),
    (_no_configs, "non-empty mapping"),
    (_bad_id, "config_id 'a|b'"),
    (_no_dimensions, "dimensions must be an object"),
])
def test_bank_validation_failures(mutate, match):
    b = make_bank()
    mutate(b)
    with pytest.raises(d2_io.BankError, match=match):
        d2_io.validate_bank(b)


@pytest.mark.parametrize("mutate", [_sum_within, _token_in_system_only])
def test_bank_variants_that_stay_valid(mutate):
    b = make_bank()
    mutate(b)
    d2_io.validate_bank(b)


def test_validation_reports_every_problem_at_once():
    b = make_bank()
    _drop_arm(b)
    _no_token(b)
    _negative_p(b)
    with pytest.raises(d2_io.BankError) as e:
        d2_io.validate_bank(b)
    msg = str(e.value)
    assert "zero_shot_cot" in msg and "occurs 0 times" in msg and "negative" in msg


def test_duplicate_json_keys_and_nan_probabilities_are_refused(tmp_path):
    p = tmp_path / "dup.json"
    raw = json.dumps(make_bank())
    p.write_text(raw.replace('"c2": {"dimensions"', '"c1": {"dimensions"', 1), encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="duplicate key"):
        d2_io.load_bank(p)
    p.write_text(raw.replace('"c1": 0.25', '"c1": NaN'), encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="finite"):
        d2_io.load_bank(p)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="valid"):
        d2_io.load_bank(p)
    with pytest.raises(d2_io.BankError, match="cannot read"):
        d2_io.load_bank(tmp_path / "absent.json")


def test_warnings_for_identical_arms_and_repeated_configurations():
    b = make_bank()
    b["configurations"]["c1"]["arms"]["zero_shot_cot"] = copy.deepcopy(b["configurations"]["c1"]["arms"]["direct_answer"])
    b["configurations"]["c3"] = copy.deepcopy(b["configurations"]["c2"])
    b["sampling_probabilities"] = {"c1": 0.25, "c2": 0.5, "c3": 0.25}
    w = d2_io.validate_bank(b)
    assert any("identical messages in both arms" in x for x in w)
    assert any("repeat the messages" in x for x in w)


# ---------------------------------------------------------------- rendering

def test_render_counts_columns_and_determinism(env):
    dec = gen.load_decoding(env.decoding)
    df = d2_io.render_requests(env.bank, ["c1", "c2", "c1"], "m1", env.items, decoding=dec)
    assert len(df) == 2 * 2 * N_ITEMS                        # the repeated c1 is rendered once
    assert list(df.columns) == list(gen.REQUEST_COLUMNS)
    assert df["request_id"].is_unique
    assert df["item_id"].dtype == "int64"
    assert set(zip(df["config_id"], df["arm"])) == {(c, a) for c in ("c1", "c2") for a in ARMS}
    assert (df["request_id"] == df["config_id"] + "|" + df["arm"] + "|" + df["item_id"].astype(str)).all()
    assert set(df["model_key"]) == {"m1"}
    again = d2_io.render_requests(env.bank, ["c1", "c2"], "m1", env.items, decoding=dec)
    pd.testing.assert_frame_equal(df, again)
    one_arm = d2_io.render_requests(env.bank, ["c1"], "m1", env.items, arms=("direct_answer",))
    assert len(one_arm) == N_ITEMS and set(one_arm["arm"]) == {"direct_answer"}
    desc = d2_io.describe_requests(df)
    assert desc["counts_by_arm"] == {"direct_answer": 10, "zero_shot_cot": 10}
    assert desc["n_configurations"] == 2 and desc["n_items"] == N_ITEMS


def test_rendered_text_is_the_skeleton_with_only_the_question_replaced(env):
    df = d2_io.render_requests(env.bank, ["c1", "c2"], "m1", env.items)
    row = df[(df.config_id == "c2") & (df.arm == "zero_shot_cot") & (df.item_id == 2)].iloc[0]
    assert json.loads(row["messages_json"]) == [
        {"role": "system", "content": "You are a careful solver."},
        {"role": "user", "content": f"Problem: {env.items['question'][2]}\nReason step by step; the last line must be 'Answer: N'."}]
    # one skeleton per (configuration, arm): replacing the question by a marker gives the same text for every item
    for (cid, arm), g in df.groupby(["config_id", "arm"]):
        texts = set()
        for mj, q in zip(g["messages_json"], env.items["question"][g["item_id"]]):
            texts.add(mj.replace(json.dumps(q, ensure_ascii=False)[1:-1], "@@"))
        assert len(texts) == 1


def test_braces_elsewhere_in_prompts_survive_and_the_question_is_not_rescanned(env):
    df = d2_io.render_requests(env.bank, ["c1"], "m1", env.items)
    d0 = df[(df.arm == "direct_answer") & (df.item_id == 0)].iloc[0]
    content = json.loads(d0["messages_json"])[0]["content"]
    for frag in ('{"a": 1}', "\\boxed{}", "{{x}}", "{answer}", "(use {braces} freely)"):
        assert frag in content
    assert "{question}" not in content
    d3 = df[(df.arm == "zero_shot_cot") & (df.item_id == 3)].iloc[0]       # its question text contains the token
    assert json.loads(d3["messages_json"])[0]["content"].count("{question}") == 1
    assert "café" in d0["messages_json"] and "\\u00e9" not in d0["messages_json"]   # ensure_ascii=False


def test_render_refusals(env):
    with pytest.raises(d2_io.BankError, match="not in the bank"):
        d2_io.render_requests(env.bank, ["c9"], "m1", env.items)
    with pytest.raises(d2_io.BankError, match="no configuration"):
        d2_io.render_requests(env.bank, [], "m1", env.items)
    with pytest.raises(d2_io.BankError, match="not unique"):
        d2_io.render_requests(env.bank, ["c1"], "m1", pd.concat([env.items, env.items.iloc[:1]]))
    with pytest.raises(d2_io.BankError, match="no items"):
        d2_io.render_requests(env.bank, ["c1"], "m1", env.items.iloc[:0])
    with pytest.raises(d2_io.BankError, match="subset"):
        d2_io.render_requests(env.bank, ["c1"], "m1", env.items, arms=("direct_answer", "few_shot"))
    with pytest.raises(d2_io.BankError, match="model key"):
        d2_io.render_requests(env.bank, ["c1"], "org/model", env.items)
    bad = copy.deepcopy(env.bank)
    _no_token(bad)
    with pytest.raises(d2_io.BankError, match="occurs 0 times"):
        d2_io.render_requests(bad, ["c1"], "m1", env.items)
    env.decoding.write_text(yaml.safe_dump(decoding_dict(max_new_tokens={"zero_shot_cot": 10})))
    with pytest.raises(d2_io.BankError, match="max_new_tokens"):
        d2_io.render_requests(env.bank, ["c1"], "m1", env.items, decoding=gen.load_decoding(env.decoding))


# ---------------------------------------------------------------- sampler link

def test_sampler_distinct_configurations_in_draw_order(env):
    bank, sha = d2_io.load_bank(env.bank_path)
    ids = d2_io.sampler_config_ids(env.sampler, bank, sha)
    s = env.sampler
    assert ids == list(dict.fromkeys(s["discovery_config_ids"] + s["validation_config_ids"]))
    assert set(ids) == {"c1", "c2"}                          # the fixed label draws both configurations


def test_sampler_drawn_from_another_bank_or_edited_is_refused(env):
    bank, sha = d2_io.load_bank(env.bank_path)
    other = copy.deepcopy(bank)
    other["configurations"]["c3"] = copy.deepcopy(other["configurations"]["c1"])
    other["sampling_probabilities"] = {"c1": 0.25, "c2": 0.5, "c3": 0.25}
    with pytest.raises(d2_io.BankError, match="not drawn from this bank"):
        d2_io.sampler_config_ids(env.sampler, other, sha)
    reweighted = copy.deepcopy(bank)
    reweighted["sampling_probabilities"] = {"c1": 0.5, "c2": 0.5}
    with pytest.raises(d2_io.BankError, match="probabilities"):
        d2_io.sampler_config_ids(env.sampler, reweighted, sha)
    edited = copy.deepcopy(env.sampler)
    edited["discovery_config_ids"][0] = "c2" if edited["discovery_config_ids"][0] == "c1" else "c1"
    with pytest.raises(d2_io.BankError, match="re-derivation"):
        d2_io.sampler_config_ids(edited, bank, sha)
    with pytest.raises(d2_io.BankError, match="bank file hash"):
        d2_io.sampler_config_ids({**env.sampler, "bank_file_sha256": "0" * 64}, bank, sha)
    with pytest.raises(d2_io.BankError, match="lacks keys"):
        d2_io.sampler_config_ids({"label": "x"}, bank, sha)


# ---------------------------------------------------------------- build_requests

def test_build_writes_one_file_per_model_and_a_record(env, capsys):
    assert env.build("m1", "m2") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "main" and summary["excluded_from_main"] is False
    bank_sha = hashlib.sha256(env.bank_path.read_bytes()).hexdigest()
    rec = env.record()
    ids = list(dict.fromkeys(env.sampler["discovery_config_ids"] + env.sampler["validation_config_ids"]))
    assert rec["bank"]["sha256"] == bank_sha and rec["bank"]["generator"] == "handwritten-test"
    assert rec["sampler"]["registered_sha256"] == env.sampler["sha256"]
    assert rec["sampler"]["file_sha256"] == hashlib.sha256(env.sampler_path.read_bytes()).hexdigest()
    assert rec["items"]["canonical_sha256"] == gsm8k.canonical_sha256(env.items)
    assert rec["items"]["n_items"] == N_ITEMS and "custom parquet" in rec["items"]["source"]
    assert rec["configuration_ids"] == ids and rec["n_draws"] == 12 and rec["n_draws_collapsed"] == 12 - len(ids)
    assert rec["excluded_from_main"] is False and rec["kind"] == "main"
    assert rec["contains_unregistered_models"] is False
    assert set(rec["models"]) == {"m1", "m2"}
    for k in ("m1", "m2"):
        e = rec["models"][k]
        assert e["n_requests"] == len(ids) * 2 * N_ITEMS
        assert e["counts_by_arm"] == {a: len(ids) * N_ITEMS for a in ARMS}
        assert e["requests_sha256"] == hashlib.sha256(env.requests(k).read_bytes()).hexdigest()
        df = gen.read_requests(env.requests(k))
        assert set(df["model_key"]) == {k} and df["request_id"].is_unique
        gen.validate_requests(df, k, gen.load_decoding(env.decoding))
    a, b = (pd.read_parquet(env.requests(k)).drop(columns="model_key") for k in ("m1", "m2"))
    pd.testing.assert_frame_equal(a, b)                       # models differ only by the model_key column


def test_build_is_idempotent_merges_models_and_refuses_changed_inputs(env):
    assert env.build("m1") == 0
    created = env.record()["created"]
    sha1 = env.record()["models"]["m1"]["requests_sha256"]
    assert env.build("m1", "m2") == 0
    rec = env.record()
    assert rec["created"] == created and set(rec["models"]) == {"m1", "m2"}
    assert rec["models"]["m1"]["requests_sha256"] == sha1
    # a different bank under the same out-dir is refused; --overwrite starts a new record
    bank = copy.deepcopy(env.bank)
    bank["configurations"]["c2"]["arms"]["direct_answer"]["messages"][1]["content"] += " Be brief."
    env.bank_path.write_text(json.dumps(bank), encoding="utf-8")
    env.bank = bank
    env.write_sampler()
    assert env.build("m1") == 2
    assert env.build("m1", extra=["--overwrite"]) == 0
    assert set(env.record()["models"]) == {"m1"}


def test_build_refuses_unregistered_models_unless_allowed(env, capsys):
    args = [a for a in env.build_args("zz_model") if a != "--allow-unregistered"]
    assert build_cli.main(args) == 2
    err = capsys.readouterr().err
    assert "zz_model" in err and "m1" in err and "m2" in err        # names the allowed keys
    assert not env.req_dir.exists()
    assert env.build("zz_model") == 0
    rec = env.record()
    assert rec["models"]["zz_model"]["registered"] is False and rec["contains_unregistered_models"] is True
    # the schema template is not a model, even when the registry holds it
    capsys.readouterr()
    assert env.build("_template") == 2
    assert d2_io.unregistered_models(["m1", "_template", "zz"], env.models) == ["_template", "zz"]


def test_build_registered_models_pass_without_the_flag(env):
    """Allowed keys need no flag; without --items-parquet the pinned GSM8K items are required."""
    args = env.build_args("m1", "m2")
    i = args.index("--items-parquet")
    args = [a for k, a in enumerate(args) if k not in (i, i + 1) and a != "--allow-unregistered"]
    if not gsm8k.is_cached("test"):
        assert build_cli.main(args) == 2
        return
    assert build_cli.main(args) == 0
    rec = env.record()
    assert rec["items"]["matches_manifest"] is True and rec["items"]["n_items"] == 1319
    assert rec["contains_unregistered_models"] is False
    assert rec["models"]["m1"]["n_requests"] == len(rec["configuration_ids"]) * 2 * 1319


def test_build_argument_refusals(env, capsys):
    assert env.build("m1", extra=["--configs", "c1"]) == 2                   # explicit list without --tranche
    assert env.build("m1", extra=["--tranche"]) == 2                         # --tranche without --configs
    assert env.build("m1", "m1") == 2                                        # duplicate keys
    assert env.build("org/model") == 2                                       # key that is not a file name
    args = [a for a in env.build_args("m1") if a != "--allow-unregistered"]
    assert build_cli.main(args) == 2                                         # --items-parquet needs --allow-unregistered
    env.sampler_path.unlink()
    assert env.build("m1") == 2                                              # sampler absent
    assert env.build("m1", extra=["--sampler", str(env.tmp / "nope.json")], sampler=False) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and not env.req_dir.exists()


def test_build_refuses_a_bank_that_is_not_frozen_or_invalid(env):
    broken = copy.deepcopy(env.bank)
    _drop_arm(broken)
    env.bank_path.write_text(json.dumps(broken), encoding="utf-8")
    assert env.build("m1") == 2
    env.bank_path.unlink()
    assert env.build("m1") == 2
    assert not env.req_dir.exists()


def test_build_refuses_arms_missing_from_the_decoding_config(env):
    env.decoding.write_text(yaml.safe_dump(decoding_dict(max_new_tokens={"direct_answer": 10})))
    assert env.build("m1") == 2
    assert not env.req_dir.exists()


def test_tranche_requests_use_an_explicit_list_and_are_marked_excluded(env):
    assert env.build("m1", extra=["--configs", "c1", "c1", "--tranche"], sampler=False) == 0
    rec = env.record(tranche=True)
    assert rec["kind"] == "tranche" and rec["excluded_from_main"] is True and rec["sampler"] is None
    assert rec["configuration_ids"] == ["c1"] and rec["n_draws"] == 2 and rec["n_draws_collapsed"] == 1
    assert env.requests("m1", tranche=True).is_file() and not env.requests("m1").exists()
    assert not (env.req_dir / "build_record.json").exists()
    assert set(pd.read_parquet(env.requests("m1", tranche=True))["config_id"]) == {"c1"}
    assert env.build("m1", extra=["--configs", "c9", "--tranche"], sampler=False) == 2     # not in the bank


def test_existing_request_file_with_other_content_needs_overwrite(env):
    assert env.build("m1") == 0
    p = env.requests("m1")
    df = pd.read_parquet(p)
    df.assign(messages_json=df["messages_json"].str.replace("Answer", "Result")).to_parquet(p, index=False)
    assert env.build("m1") == 2
    assert env.build("m1", extra=["--overwrite"]) == 0
    assert "Answer" in pd.read_parquet(p)["messages_json"].iloc[0]


# ---------------------------------------------------------------- compatibility with the other consumers

# ---------------------------------------------------------------- scorer manifests

def _write_scorer(path, name="answer_colon", **over):
    rec = scoring.scorer_record(scoring.CANDIDATE_SPECS[name])
    rec.update(over)
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_registered_scorer_manifest(tmp_path):
    spec, rec = d2_io.load_registered_scorer(_write_scorer(tmp_path / "s.json"))
    assert spec == scoring.CANDIDATE_SPECS["answer_colon"] and rec["status"] == "registered"
    assert rec["scorer_version"] == scoring.SCORER_VERSION and len(rec["manifest_sha256"]) == 64
    alias = json.loads((tmp_path / "s.json").read_text())
    alias["spec"] = alias.pop("format_spec")                  # the key src/gate.py also accepts
    (tmp_path / "alias.json").write_text(json.dumps(alias))
    assert d2_io.load_registered_scorer(tmp_path / "alias.json")[0] == spec
    with pytest.raises(d2_io.ScoreInputError, match="does not exist"):
        d2_io.load_registered_scorer(tmp_path / "absent.json")
    with pytest.raises(d2_io.ScoreInputError, match="scorer_version"):
        d2_io.load_registered_scorer(_write_scorer(tmp_path / "v.json", scorer_version="gsm8k-scorer/0.0.1"))
    with pytest.raises(d2_io.ScoreInputError, match="format_spec_sha256"):
        d2_io.load_registered_scorer(_write_scorer(tmp_path / "f.json", format_spec_sha256="0" * 64))
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(d2_io.ScoreInputError, match="format_spec"):
        d2_io.load_registered_scorer(tmp_path / "bad.json")
    with pytest.raises(d2_io.ScoreInputError, match="unknown candidate"):
        d2_io.candidate_scorer("nope")
    assert d2_io.candidate_scorer("hash_n")[1]["status"] == "candidate, not registered"


def test_score_refuses_an_absent_scorer_before_reading_anything(env, capsys):
    assert env.score("nowhere", extra=["--scorer", str(env.tmp / "absent.json")]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert env.score("nowhere", extra=["--candidate-spec", "nope"]) == 2
    assert env.score("nowhere", extra=["--scorer", str(_write_scorer(env.tmp / "s.json"))]) == 2   # --items-parquet needs a candidate
    assert not (env.tmp / "scored.parquet").exists()


# ---------------------------------------------------------------- end to end

def _ids(env):
    return list(dict.fromkeys(env.sampler["discovery_config_ids"] + env.sampler["validation_config_ids"]))


def test_end_to_end_dry_run(env, capsys):
    assert env.build("m1", "m2") == 0
    env.generate("m1", env.tmp / "g" / "m1")                          # plain EchoBackend: no extractable answers
    env.generate("m2", env.tmp / "g" / "m2", backend="gold")          # outputs end in 'Answer: N', half of them right
    capsys.readouterr()
    out = env.tmp / "scored" / "scored.parquet"
    assert env.score(env.tmp / "g" / "m1", env.tmp / "g" / "m2", out=out) == 0
    cap = capsys.readouterr()
    assert "candidate, not registered" in cap.err
    res = json.loads(cap.out)
    assert res["scorer_status"] == "candidate, not registered" and res["all_requests_matched"] is True
    scored = pd.read_parquet(out)
    assert list(scored.columns) == list(d2_io.SCORED_COLUMNS)
    ids = _ids(env)
    assert len(scored) == 2 * len(ids) * 2 * N_ITEMS
    assert not scored.duplicated(["model_key", "arm", "config_id", "item_id"]).any()
    assert scored["item_id"].dtype == "int64" and scored["compliant"].dtype == bool
    # independent check against the raw outputs
    for key in ("m1", "m2"):
        raw = gen.read_outputs(env.tmp / "g" / key).merge(env.items[["item_id", "gold_answer"]], on="item_id")
        raw["expect"] = [int(t.rstrip().endswith(f"Answer: {g}")) for t, g in zip(raw.output_text, raw.gold_answer)]
        got = scored[scored.model_key == key].merge(raw[["arm", "config_id", "item_id", "expect", "cached"]],
                                                    on=["arm", "config_id", "item_id"], suffixes=("", "_raw"))
        assert len(got) == len(raw)
        assert (got["strict_correct"] == got["expect"]).all() and (got["cached"] == got["cached_raw"]).all()
    m1 = scored[scored.model_key == "m1"]
    m2 = scored[scored.model_key == "m2"]
    assert m1["strict_correct"].sum() == 0 and not m1["compliant"].any()
    assert m2["compliant"].all() and 0 < m2["strict_correct"].sum() < len(m2)
    assert (m2["lenient_correct"] >= m2["strict_correct"]).all()
    # the analysis adapter accepts the table, including the filtered read that scripts/run_dataset2.py uses
    filtered = pd.read_parquet(out, filters=[("config_id", "in", sorted(ids))])
    pd.testing.assert_frame_equal(filtered.reset_index(drop=True), scored)
    cells = d2.cell_table(scored, ["m1", "m2"], ids, env.items["item_id"].tolist())
    assert len(cells) == 2 * 2 * len(ids) and (cells["n_items"] == N_ITEMS).all()
    B, P = d2.unit_arrays(cells, env.sampler["discovery_config_ids"], ["m1", "m2"])
    assert B.shape == P.shape == (6, 2)
    # provenance: parquet key-value metadata and the sidecar
    md = pq.read_schema(out).metadata
    assert md[b"d2_scorer_status"] == b"candidate, not registered" and md[b"d2_kind"] == b"main"
    assert md[b"d2_excluded_from_main"] == b"false"
    assert json.loads(md[b"d2_scorer_record"])["format_spec"]["name"] == "answer_colon"
    inputs = json.loads(md[b"d2_inputs"])
    assert {i["model_key"] for i in inputs} == {"m1", "m2"} and all(i["test_backend"] for i in inputs)
    rec = env.record()
    assert {i["bank_sha256"] for i in inputs} == {rec["bank"]["sha256"]}
    assert {tuple(i["requests_sha256"]) for i in inputs} == {(rec["models"][k]["requests_sha256"],) for k in ("m1", "m2")}
    side = json.loads((env.tmp / "scored" / "scored.meta.json").read_text(encoding="utf-8"))
    assert side["scorer"]["status"] == "candidate, not registered" and side["n_rows"] == len(scored)
    assert side["scored_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert side["scorer"]["scorer_version"] == scoring.SCORER_VERSION
    assert {s["model_key"] for s in side["summary_by_model_arm"]} == {"m1", "m2"}
    # no accuracy before the prediction rules are saved (before validation is scored): the table pools validation units
    assert not any("correct" in k for s in side["summary_by_model_arm"] for k in s)
    assert "hard rule 6" in side["summary_note"]
    assert json.loads(md[b"d2_allow_flags"]) == {"allow_test_backend": True, "allow_incomplete": False,
                                                 "allow_unmatched_requests": False}
    assert all(i["incomplete_allowed"] is False for i in inputs)
    assert side["inputs"][0]["runs"][0]["status"] == "complete" and "pandas" in side["libraries"]
    # a second scoring run reproduces the table
    out2 = env.tmp / "scored2.parquet"
    assert env.score(env.tmp / "g" / "m2", env.tmp / "g" / "m1", out=out2) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(out2), scored)


def test_registered_scorer_labels_the_table_registered(env, monkeypatch):
    # The pinned items are needed with a fixed scorer, so the loader is replaced by the fabricated items.
    assert env.build("m1") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    scorer = _write_scorer(env.tmp / "gsm8k_scorer.json")
    monkeypatch.setattr(d2_io, "load_items", lambda p=None, manifest_path=None, split="test": (
        env.items, {"source": "fabricated", "canonical_sha256": "x", "n_items": N_ITEMS}))
    out = env.tmp / "reg.parquet"
    rc = score_cli.main(["--work-dir", str(env.tmp / "g" / "m1"), "--scorer", str(scorer), "--out", str(out),
                         "--requests-dir", str(env.req_dir), "--allow-test-backend"])
    assert rc == 0
    md = pq.read_schema(out).metadata
    assert md[b"d2_scorer_status"] == b"registered"
    rec = json.loads(md[b"d2_scorer_record"])
    assert rec["manifest_sha256"] == hashlib.sha256(scorer.read_bytes()).hexdigest()


def test_tranche_and_main_outputs_are_never_mixed(env, capsys):
    assert env.build("m1") == 0
    assert env.build("m1", extra=["--configs", "c1", "--tranche"], sampler=False) == 0
    env.generate("m1", env.tmp / "g_conf" / "m1", backend="gold")
    env.generate("m1", env.tmp / "g_tr" / "m1", tranche=True, backend="gold")
    capsys.readouterr()
    assert env.score(env.tmp / "g_tr" / "m1") == 2                                  # tranche folder, no --tranche
    assert "tranche" in capsys.readouterr().err
    flags = ("--candidate-spec", "answer_colon", "--allow-test-backend", "--tranche")
    assert env.score(env.tmp / "g_conf" / "m1", extra=flags) == 2                   # main folder with --tranche
    out = env.tmp / "tr" / "tranche_scored.parquet"
    assert env.score(env.tmp / "g_tr" / "m1", out=out, extra=flags) == 0
    md = pq.read_schema(out).metadata
    assert md[b"d2_kind"] == b"tranche" and md[b"d2_excluded_from_main"] == b"true"
    scored = pd.read_parquet(out)
    cells = d2.cell_table(scored, ["m1"], ["c1"], env.items["item_id"].tolist())    # what precision_matched reads
    assert len(cells) == 2 and set(scored["config_id"]) == {"c1"}
    # the tranche default output is a different path from the main default
    assert score_cli.DEFAULT_TRANCHE_OUT != score_cli.DEFAULT_OUT and "tranche" in score_cli.DEFAULT_TRANCHE_OUT
    assert env.score(env.tmp / "g_tr" / "m1", out=score_cli.DEFAULT_OUT, extra=flags) == 2   # never into the main table


def test_throughput_folders_are_refused(env, capsys):
    assert env.build("m1") == 0
    work = env.tmp / "g" / "m1"
    env.generate("m1", work, backend="gold", throughput=True, limit=8)
    tdir = work / "throughput"
    assert any(tdir.glob("throughput_log-*.json")) and any(tdir.glob("shard-*.parquet"))
    capsys.readouterr()
    assert env.score(tdir) == 2
    assert "throughput" in capsys.readouterr().err
    # the same shards under another folder name, logs removed: the shard metadata still marks them
    copy_dir = env.tmp / "copied"
    copy_dir.mkdir()
    for p in tdir.glob("shard-*.parquet"):
        shutil.copy(p, copy_dir / p.name)
    assert env.score(copy_dir) == 2
    assert "throughput shard" in capsys.readouterr().err
    # throughput logs next to ordinary shards
    mixed = env.tmp / "mixed"
    mixed.mkdir()
    env.generate("m1", env.tmp / "g2" / "m1", backend="gold")
    for p in (env.tmp / "g2" / "m1").iterdir():
        shutil.copy(p, mixed / p.name)
    shutil.copy(next(tdir.glob("throughput_log-*.json")), mixed / "throughput_log-x.json")
    assert env.score(mixed) == 2
    assert "throughput logs" in capsys.readouterr().err
    # a run log that says throughput
    flagged = env.tmp / "flagged"
    shutil.copytree(env.tmp / "g2" / "m1", flagged)
    lp = next(flagged.glob("run_log-*.json"))
    log = json.loads(lp.read_text())
    log["mode"], log["excluded_from_main"] = "throughput", True
    lp.write_text(json.dumps(log))
    assert env.score(flagged) == 2
    assert "throughput-mode log" in capsys.readouterr().err
    # the main out-dir of a run that also did a throughput run is scoreable; the subfolder is not read
    assert env.generate("m1", work, backend="gold")["status"] == "complete"
    assert env.score(work) == 0
    assert pd.read_parquet(env.tmp / "scored.parquet")["model_key"].nunique() == 1


def test_other_refusals(env, capsys):
    assert env.build("m1", "m2") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    g1 = env.tmp / "g" / "m1"
    capsys.readouterr()
    # test backend needs its flag
    assert env.score(g1, extra=["--candidate-spec", "answer_colon"]) == 2
    assert "test backend" in capsys.readouterr().err
    # the same model in two folders would duplicate rows
    shutil.copytree(g1, env.tmp / "g" / "copy")
    assert env.score(g1, env.tmp / "g" / "copy") == 2
    assert "two folders" in capsys.readouterr().err
    # a folder without a run log, an empty folder, a missing folder
    nolog = env.tmp / "nolog"
    shutil.copytree(g1, nolog)
    for p in nolog.glob("run_log-*.json"):
        p.unlink()
    assert env.score(nolog) == 2 and "run_log" in capsys.readouterr().err
    (env.tmp / "empty").mkdir()
    assert env.score(env.tmp / "empty") == 2
    assert env.score(env.tmp / "missing") == 2
    # a request file with no build record
    assert env.score(g1, extra=["--candidate-spec", "answer_colon", "--allow-test-backend",
                                "--requests-dir", str(env.tmp / "elsewhere")]) == 2
    assert "build record" in capsys.readouterr().err
    assert env.score(g1, out=env.tmp / "diag.parquet", extra=["--candidate-spec", "answer_colon", "--allow-test-backend",
                                                               "--requests-dir", str(env.tmp / "elsewhere"),
                                                               "--allow-unmatched-requests"]) == 0
    assert json.loads(capsys.readouterr().out)["all_requests_matched"] is False
    # a pilot (--limit) or interrupted run is not a complete table
    env.generate("m2", env.tmp / "g" / "m2", backend="gold", limit=7)
    assert env.score(env.tmp / "g" / "m2") == 2
    assert "no complete run without --limit" in capsys.readouterr().err
    assert env.score(env.tmp / "g" / "m2", out=env.tmp / "part.parquet",
                     extra=["--candidate-spec", "answer_colon", "--allow-test-backend", "--allow-incomplete"]) == 0
    part = pd.read_parquet(env.tmp / "part.parquet")
    with pytest.raises(d2.IncompleteData):                    # the analysis adapter still refuses partial data
        d2.cell_table(part, ["m2"], _ids(env), env.items["item_id"].tolist())
    # resuming the same folder to completion makes it scoreable
    env.generate("m2", env.tmp / "g" / "m2", backend="gold")
    assert env.score(env.tmp / "g" / "m2") == 0


def test_outputs_for_items_without_a_gold_answer_are_refused(env, capsys):
    assert env.build("m1") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    short = env.items.iloc[:3]
    short.to_parquet(env.items_path, index=False)
    capsys.readouterr()
    assert env.score(env.tmp / "g" / "m1") == 2
    assert "gold" in capsys.readouterr().err


def test_duplicated_rows_across_folders_are_refused(env):
    assert env.build("m1") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    out, info = d2_io.inspect_work_dir(env.tmp / "g" / "m1", allow_test_backend=True)
    spec, _ = d2_io.candidate_scorer("answer_colon")
    gold = gsm8k.gold_lookup(env.items)
    table, _ = d2_io.score_work_dirs([env.tmp / "g" / "m1"], spec, gold, records=d2_io.load_build_records(env.req_dir),
                                     tranche=False, allow_test_backend=True)
    assert not table.duplicated(d2_io.KEY_COLUMNS).any() and len(table) == len(out)


def test_write_scored_roundtrip_keeps_dtypes_and_adds_metadata(tmp_path):
    df = pd.DataFrame({"model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": pd.Series([1], dtype="int64"),
                       "strict_correct": [1], "lenient_correct": [1], "compliant": [True], "finish_reason": ["stop"],
                       "n_output_tokens": [3], "cached": [False], "prompt_sha256": ["0" * 64]})
    p = d2_io.write_scored(df, tmp_path / "x" / "t.parquet", {"d2_note": "hello"})
    pd.testing.assert_frame_equal(pd.read_parquet(p), df)
    assert pq.read_schema(p).metadata[b"d2_note"] == b"hello"
    assert d2_io.sidecar_path(p).name == "t.meta.json"


# ---------------------------------------------------------------- pinned items (data dependent)

@pytest.mark.skipif(not gsm8k.is_cached("test") or not Path(gsm8k.MANIFEST_PATH).exists(),
                    reason="data/raw/gsm8k test split or manifests/gsm8k_items.json absent")
def test_pinned_items_match_the_items_manifest_and_render_at_full_size(env):
    items, info = d2_io.load_items()
    assert info["matches_manifest"] is True and info["n_items"] == 1319
    assert info["canonical_sha256"] == info["manifest_canonical_sha256"]
    df = d2_io.render_requests(env.bank, ["c1"], "m1", items, decoding=gen.load_decoding(env.decoding))
    assert len(df) == 2 * 1319 and df["request_id"].is_unique
    wrong = env.tmp / "items_manifest.json"
    wrong.write_text(json.dumps({"splits": {"test": {"canonical_sha256": "0" * 64}}}))
    with pytest.raises(d2_io.ScoreInputError, match="differ"):
        d2_io.load_items(manifest_path=wrong)


# ---------------------------------------------------------------- splits: train items, calibration sample, train builds

needs_train = pytest.mark.skipif(not gsm8k.is_cached("train") or not Path(gsm8k.MANIFEST_PATH).exists(),
                                 reason="data/raw/gsm8k train split or manifests/gsm8k_items.json absent")


@needs_train
def test_pinned_train_items_match_the_items_manifest(env):
    items, info = d2_io.load_items(split="train")
    assert info["split"] == "train" and info["n_items"] == 7473 and info["matches_manifest"] is True
    assert info["canonical_sha256"] == info["manifest_canonical_sha256"]
    assert "train" in info["source"] and info["path"].endswith("train-00000-of-00001.parquet")
    assert items["item_id"].tolist() == list(range(7473)) and items["gold_answer"].map(bool).all()
    if gsm8k.is_cached("test"):
        test_items, test_info = d2_io.load_items()
        assert test_info["split"] == "test" and test_info["canonical_sha256"] != info["canonical_sha256"]
    wrong = env.tmp / "wrong_manifest.json"
    wrong.write_text(json.dumps({"splits": {"train": {"canonical_sha256": "0" * 64}}}))
    with pytest.raises(d2_io.ScoreInputError, match="train items differ"):
        d2_io.load_items(manifest_path=wrong, split="train")
    only_test = env.tmp / "only_test.json"
    only_test.write_text(json.dumps({"splits": {"test": {"canonical_sha256": "0" * 64}}}))
    with pytest.raises(d2_io.ScoreInputError, match="cannot read the items manifest"):
        d2_io.load_items(manifest_path=only_test, split="train")


def test_split_argument_is_validated_and_labels_custom_parquets(env):
    with pytest.raises(d2_io.ScoreInputError, match="split must be"):
        d2_io.load_items(split="dev")
    with pytest.raises(d2_io.ScoreInputError, match="split must be"):
        d2_io.load_items(env.items_path, split="validation")
    items, info = d2_io.load_items(env.items_path, split="train")
    assert info["split"] == "train" and "train split" in info["source"] and info["matches_manifest"] is None
    assert d2_io.load_items(env.items_path)[1]["split"] == "test"
    assert d2_io.SPLITS == ("test", "train")


def test_calibration_sample_is_deterministic_sorted_and_drawn_from_ids_only():
    ids = list(range(0, 400, 2))
    a = d2_io.calibration_sample(ids, 25)
    assert a == d2_io.calibration_sample(list(reversed(ids)), 25)             # the order of the ids is irrelevant
    assert a == sorted(a) and len(set(a)) == 25 and set(a) <= set(ids)
    expect = sorted(int(i) for i in sampling.rng_for("gsm8k-calibration-items", 25).choice(
        np.asarray(ids), size=25, replace=False))
    assert a == expect
    assert a != d2_io.calibration_sample(ids, 26)[:25]                        # another N is a separate draw
    assert d2_io.calibration_sample(ids, len(ids)) == ids                     # N = population returns everything
    for bad in (0, -1, len(ids) + 1, True, 2.5):
        with pytest.raises(d2_io.BankError, match="between 1 and"):
            d2_io.calibration_sample(ids, bad)


@needs_train
def test_calibration_sample_on_the_real_train_split_is_stable():
    ids = gsm8k.load_gsm8k("train")["item_id"].tolist()
    a = d2_io.calibration_sample(ids, 200)
    assert a == d2_io.calibration_sample(ids, 200) and len(a) == 200 and 0 <= a[0] and a[-1] < 7473
    assert len(set(a)) == 200


def _fab_items(n=12, gold_shift=0):
    items = make_items(n)
    items["gold_answer"] = [str(2 * i + gold_shift) for i in range(n)]
    info = {"split": "train", "source": "fabricated", "canonical_sha256": gsm8k.canonical_sha256(items), "n_items": n}
    return items, info


def test_calibration_manifest_is_written_once_reused_and_refuses_a_different_n(tmp_path):
    items, info = _fab_items()
    path = tmp_path / "cal.json"
    sub, i1, rec = d2_io.select_calibration_items(items, info, 5, path)
    assert rec["status"] == "written" and path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["split"] == "train" and body["n_items"] == 5 and body["seed_label"] == "gsm8k-calibration-items"
    assert body["revision"] == gsm8k.REVISION and body["dataset"] == gsm8k.DATASET
    assert body["item_ids"] == sorted(body["item_ids"]) == sub["item_id"].tolist() and len(body["item_ids"]) == 5
    assert len(body["sha256"]) == 64 and "status" not in body and body["rng_labels"] == ["gsm8k-calibration-items", 5]
    assert i1["calibration_items"]["sha256"] == body["sha256"] and i1["n_items"] == 5 and i1["population_n_items"] == 12
    assert i1["canonical_sha256"] == gsm8k.canonical_sha256(sub) != info["canonical_sha256"]
    before = path.read_bytes()
    _, _, again = d2_io.select_calibration_items(items, info, 5, path)
    assert again["status"] == "reused" and path.read_bytes() == before
    with pytest.raises(d2_io.BankError, match="holds 5 items, not the requested 6"):
        d2_io.select_calibration_items(items, info, 6, path)
    assert path.read_bytes() == before                                         # refusal leaves the sample alone
    _, _, over = d2_io.select_calibration_items(items, info, 6, path, overwrite=True)
    assert over["status"] == "overwritten" and json.loads(path.read_text())["n_items"] == 6
    with pytest.raises(d2_io.BankError, match="train"):
        d2_io.select_calibration_items(items, {**info, "split": "test"}, 5, tmp_path / "x.json")


def test_calibration_manifest_edits_and_other_populations_are_refused(tmp_path):
    items, info = _fab_items()
    path = tmp_path / "cal.json"
    d2_io.select_calibration_items(items, info, 5, path)
    good = json.loads(path.read_text(encoding="utf-8"))
    edited = dict(good, item_ids=[good["item_ids"][0] + 1] + good["item_ids"][1:])
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="sha256 does not match"):
        d2_io.select_calibration_items(items, info, 5, path)
    edited["sha256"] = d2_io._calibration_body_sha(edited)                      # a careful edit still differs from the draw
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="deterministic draw"):
        d2_io.select_calibration_items(items, info, 5, path)
    path.write_text(json.dumps(good), encoding="utf-8")
    other, other_info = _fab_items(gold_shift=1)                               # same ids, other train items
    with pytest.raises(d2_io.BankError, match="different train items"):
        d2_io.select_calibration_items(other, other_info, 5, path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(d2_io.BankError, match="not readable JSON"):
        d2_io.select_calibration_items(items, info, 5, path)
    _, _, rec = d2_io.select_calibration_items(items, info, 5, path, overwrite=True)
    assert rec["status"] == "overwritten" and json.loads(path.read_text())["item_ids"] == good["item_ids"]


def test_calibration_sample_depends_on_item_ids_only(tmp_path):
    a, ia = _fab_items()
    b, ib = _fab_items(gold_shift=7)
    b["question"] = [q[::-1] for q in b["question"]]
    ib["canonical_sha256"] = gsm8k.canonical_sha256(b)
    s1 = d2_io.select_calibration_items(a, ia, 5, tmp_path / "a.json")[2]["item_ids"]
    s2 = d2_io.select_calibration_items(b, ib, 5, tmp_path / "b.json")[2]["item_ids"]
    assert s1 == s2


def test_request_ids_never_repeat_across_splits(env):
    test_df = d2_io.render_requests(env.bank, ["c1", "c2"], "m1", env.items)
    train_df = d2_io.render_requests(env.bank, ["c1", "c2"], "m1", env.items, split="train")
    assert not set(test_df["request_id"]) & set(train_df["request_id"])
    assert train_df["request_id"].is_unique and train_df["request_id"].str.startswith("train|").all()
    assert not test_df["request_id"].str.startswith("train|").any()
    assert (train_df["request_id"] == "train|" + train_df["config_id"] + "|" + train_df["arm"] + "|"
            + train_df["item_id"].astype(str)).all()
    assert test_df["messages_json"].tolist() == train_df["messages_json"].tolist()   # same skeleton, other id namespace
    assert d2_io.request_id_for("c1", "direct_answer", 7) == "c1|direct_answer|7"
    assert d2_io.request_id_for("c1", "direct_answer", 7, "train") == "train|c1|direct_answer|7"
    with pytest.raises(d2_io.ScoreInputError, match="split must be"):
        d2_io.render_requests(env.bank, ["c1"], "m1", env.items, split="dev")
    both = pd.concat([test_df, train_df])
    assert both["request_id"].is_unique                                          # one frame can hold both splits


def test_train_screening_build_goes_under_screening_and_is_marked(env, capsys):
    assert env.build_train("m1", "m2") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "screening" and out["split"] == "train" and out["purpose"] == "admissibility_screening"
    assert out["calibration_items"]["status"] == "written" and out["excluded_from_main"] is True
    rec = env.record(sub="screening")
    assert rec["kind"] == "screening" and rec["split"] == "train" and rec["purpose"] == "admissibility_screening"
    assert rec["excluded_from_main"] is True and rec["sampler"] is None
    assert rec["configuration_ids"] == ["c1", "c2"]                              # every configuration of the bank
    cal = json.loads(env.cal_path.read_text(encoding="utf-8"))
    assert rec["items"]["calibration_items"]["sha256"] == cal["sha256"] and rec["items"]["n_items"] == 5
    assert rec["items"]["split"] == "train" and rec["items"]["population_n_items"] == 12
    assert (env.req_dir / "screening" / "m1.parquet").is_file() and (env.req_dir / "screening" / "m2.parquet").is_file()
    assert not (env.req_dir / "build_record.json").exists() and not (env.req_dir / "tranche").exists()
    df = pd.read_parquet(env.requests("m1", sub="screening"))
    assert len(df) == 2 * 2 * 5 and set(df["item_id"]) == set(cal["item_ids"])
    assert df["request_id"].str.startswith("train|").all() and df["request_id"].is_unique
    gen.validate_requests(df, "m1", gen.load_decoding(env.decoding))
    row = df[(df["config_id"] == "c2") & (df["arm"] == "direct_answer")].iloc[0]
    assert env.items["question"][int(row["item_id"])] in json.loads(row["messages_json"])[-1]["content"]
    assert rec["models"]["m1"]["requests_sha256"] == hashlib.sha256(env.requests("m1", sub="screening").read_bytes()).hexdigest()
    # a rerun reuses the sample and leaves the files alone; other inputs need --overwrite
    assert env.build_train("m1") == 0
    assert json.loads(capsys.readouterr().out)["calibration_items"]["status"] == "reused"
    assert env.build_train("m1", extra=["--configs", "c2"]) == 2
    assert env.build_train("m1", extra=["--configs", "c2", "--overwrite"]) == 0
    assert env.record(sub="screening")["configuration_ids"] == ["c2"]


def test_a_different_n_is_refused_until_overwrite(env, capsys):
    assert env.build_train("m1") == 0
    before = env.cal_path.read_bytes()
    capsys.readouterr()
    assert env.build_train("m1", n=6) == 2
    assert "cannot be reused" in capsys.readouterr().err and env.cal_path.read_bytes() == before
    assert env.build_train("m1", n=6, extra=["--overwrite"]) == 0
    assert json.loads(env.cal_path.read_text())["n_items"] == 6
    assert env.record(sub="screening")["items"]["n_items"] == 6


def test_train_tranche_build_goes_under_tranche_and_is_marked(env):
    assert env.build_train("m1", mode="--tranche", extra=["--configs", "c1"]) == 0
    rec = env.record(tranche=True)
    assert rec["kind"] == "tranche" and rec["split"] == "train" and rec["purpose"] == "throughput_noise_tranche"
    assert rec["excluded_from_main"] is True and rec["configuration_ids"] == ["c1"] and rec["sampler"] is None
    assert not (env.req_dir / "screening").exists() and not (env.req_dir / "build_record.json").exists()
    assert pd.read_parquet(env.requests("m1", tranche=True))["request_id"].str.startswith("train|").all()
    assert env.build_train("m1") == 0                                            # screening keeps its own record
    assert env.record(sub="screening")["kind"] == "screening" and env.record(tranche=True)["kind"] == "tranche"
    # a tranche on the test items cannot silently join a train tranche record
    assert env.build("m1", extra=["--configs", "c1", "--tranche"], sampler=False) == 2


def test_test_tranche_warns_that_it_observes_target_items(env, capsys):
    assert env.build("m1", extra=["--configs", "c1", "--tranche"], sampler=False) == 0
    err = capsys.readouterr().err
    assert "test items" in err and "--split train" in err
    assert env.record(tranche=True)["split"] == "test" and env.record(tranche=True)["purpose"] == "throughput_noise_tranche"


def test_main_records_say_test_and_are_not_excluded(env):
    assert env.build("m1") == 0
    rec = env.record()
    assert rec["split"] == "test" and rec["purpose"] == "main_collection" and rec["excluded_from_main"] is False
    assert rec["items"]["split"] == "test"
    assert pd.read_parquet(env.requests("m1"))["request_id"].str.startswith("train|").sum() == 0


@pytest.mark.parametrize("extra, msg", [
    (["--split", "train", "--calibration-items", "5"], "add --screening"),
    (["--screening"], "add --split train"),
    (["--split", "test", "--screening"], "add --split train"),
    (["--split", "train", "--screening"], "needs --calibration-items"),
    (["--calibration-items", "5", "--tranche", "--configs", "c1"], "train only"),
    (["--split", "train", "--calibration-items", "5", "--screening", "--tranche", "--configs", "c1"], "separate builds"),
    (["--split", "train", "--calibration-items", "5", "--tranche"], "needs --configs"),
    (["--configs", "c1"], "only for a tranche or screening"),
    (["--split", "train", "--calibration-items", "0", "--screening", "--calibration-manifest", "{cal}"], "between 1 and"),
    (["--split", "train", "--calibration-items", "13", "--screening", "--calibration-manifest", "{cal}"], "between 1 and"),
    (["--split", "train", "--calibration-items", "5", "--screening"], "explicit --calibration-manifest"),
])
def test_train_build_refusals_write_nothing(env, capsys, monkeypatch, extra, msg):
    env.use_train_items()
    repo_manifest = env.tmp / "repo_calibration_items.json"
    monkeypatch.setattr(d2_io, "CALIBRATION_MANIFEST", repo_manifest)
    extra = [str(env.cal_path) if a == "{cal}" else a for a in extra]
    assert env.build("m1", extra=extra, sampler=False) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and msg in err
    assert not env.req_dir.exists() and not repo_manifest.exists() and not env.cal_path.exists()


# ---------------------------------------------------------------- scoring train outputs

def _generated_screening_env(env, key="m1"):
    assert env.build_train(key) == 0
    env.generate(key, env.tmp / "g" / key, backend="gold", sub="screening")
    return env.tmp / "g" / key


def test_train_screening_outputs_are_scored_into_the_screening_table(env, capsys):
    g = _generated_screening_env(env)
    capsys.readouterr()
    out = env.tmp / "scored" / "screening_scored.parquet"
    flags = ("--candidate-spec", "answer_colon", "--allow-test-backend", "--screening")
    assert env.score(g, out=out, extra=flags) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["kind"] == "screening" and res["split"] == "train" and res["all_requests_matched"] is True
    md = pq.read_schema(out).metadata
    assert md[b"d2_kind"] == b"screening" and md[b"d2_split"] == b"train" and md[b"d2_excluded_from_main"] == b"true"
    assert md[b"d2_purpose"] == b"admissibility_screening"
    scored = pd.read_parquet(out)
    cal = json.loads(env.cal_path.read_text(encoding="utf-8"))
    assert set(scored["item_id"]) == set(cal["item_ids"]) and len(scored) == 2 * 2 * 5
    # no correctness in a screening table or its summary
    assert list(scored.columns) == list(d2_io.SCREENING_COLUMNS)
    assert not any(c in scored.columns for c in d2_io.CORRECTNESS_COLUMNS)
    inp = json.loads(md[b"d2_inputs"])[0]
    assert inp["split"] == "train" and inp["request_kind"] == "screening"
    assert inp["calibration_items"]["sha256"] == cal["sha256"]
    side = json.loads(d2_io.sidecar_path(out).read_text(encoding="utf-8"))
    assert side["split"] == "train" and side["kind"] == "screening" and side["excluded_from_main"] is True
    assert not any("correct" in k for r in side["summary_by_model_arm"] for k in r) and "hard rule 2" in side["units"]
    defaults = {score_cli.DEFAULT_OUT, score_cli.DEFAULT_TRANCHE_OUT, score_cli.DEFAULT_SCREENING_OUT}
    assert len(defaults) == 3 and score_cli.DEFAULT_SCREENING_OUT.endswith("screening_scored.parquet")
    assert score_cli.DEFAULT_BY_KIND["screening"] == score_cli.DEFAULT_SCREENING_OUT


def test_scoring_uses_the_gold_answers_of_the_requests_split(env, capsys, monkeypatch):
    assert env.build_train("m1", mode="--tranche", extra=("--configs", "c1")) == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold", sub="tranche")
    g = env.tmp / "g" / "m1"
    train_items, test_items = env.items, env.items.assign(gold_answer="999")      # a gold no output carries
    calls = []

    def fake_load_items(p=None, manifest_path=None, split="test"):
        calls.append(split)
        return (train_items if split == "train" else test_items), {
            "source": f"fabricated {split}", "split": split, "canonical_sha256": "x", "n_items": len(train_items)}

    monkeypatch.setattr(d2_io, "load_items", fake_load_items)
    out = env.tmp / "gold" / "s.parquet"
    rc = score_cli.main(["--work-dir", str(g), "--candidate-spec", "answer_colon", "--allow-test-backend", "--tranche",
                         "--out", str(out), "--requests-dir", str(env.req_dir)])
    assert rc == 0 and calls == ["train"]
    assert pd.read_parquet(out)["strict_correct"].sum() > 0
    capsys.readouterr()
    rc = score_cli.main(["--work-dir", str(g), "--candidate-spec", "answer_colon", "--allow-test-backend", "--tranche",
                         "--split", "train", "--out", str(env.tmp / "gold" / "t.parquet"), "--requests-dir", str(env.req_dir)])
    assert rc == 0 and calls == ["train", "train"]


def test_split_and_kind_mismatches_are_refused_when_scoring(env, capsys):
    g = _generated_screening_env(env)
    base = ("--candidate-spec", "answer_colon", "--allow-test-backend")
    out = env.tmp / "never.parquet"
    capsys.readouterr()
    # train outputs are never a main table
    assert env.score(g, out=out, extra=base) == 2
    assert "main table" in capsys.readouterr().err
    # a screening folder scored as a tranche, or with a contradicting split, or with both kinds
    assert env.score(g, out=out, extra=base + ("--tranche",)) == 2
    err = capsys.readouterr().err
    assert "built as 'screening'" in err and "--screening" in err
    assert env.score(g, out=out, extra=base + ("--screening", "--split", "test")) == 2
    assert "contradicts" in capsys.readouterr().err
    assert env.score(g, out=out, extra=base + ("--screening", "--tranche")) == 2
    assert "separate kinds" in capsys.readouterr().err
    assert not out.exists()
    # a screening table never goes to another kind's default file
    for other in (score_cli.DEFAULT_OUT, score_cli.DEFAULT_TRANCHE_OUT):
        assert env.score(g, out=other, extra=base + ("--screening",)) == 2
        assert "must not go there" in capsys.readouterr().err
    # without a build record the split must be stated, and train still needs a kind
    elsewhere = ("--requests-dir", str(env.tmp / "elsewhere"), "--allow-unmatched-requests")
    assert env.score(g, out=out, extra=base + elsewhere + ("--split", "train")) == 2
    assert "main table" in capsys.readouterr().err
    assert not out.exists()
    assert env.score(g, out=out, extra=base + elsewhere + ("--split", "train", "--screening")) == 0
    assert pq.read_schema(out).metadata[b"d2_split"] == b"train"


def test_test_folder_is_refused_with_a_train_split_and_mixed_splits_are_refused(env, capsys):
    assert env.build("m1") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    capsys.readouterr()
    flags = ("--candidate-spec", "answer_colon", "--allow-test-backend")
    assert env.score(env.tmp / "g" / "m1", extra=flags + ("--split", "train")) == 2
    assert "contradicts" in capsys.readouterr().err
    assert env.score(env.tmp / "g" / "m1", extra=flags + ("--screening",)) == 2
    assert "built as 'main'" in capsys.readouterr().err
    # a train folder next to a test folder
    env.use_train_items()
    assert env.build_train("m2") == 0
    env.generate("m2", env.tmp / "g" / "m2", backend="gold", sub="screening")
    capsys.readouterr()
    assert env.score(env.tmp / "g" / "m1", env.tmp / "g" / "m2", extra=flags + ("--screening",)) == 2
    assert "different splits" in capsys.readouterr().err
    recs = d2_io.load_build_records(env.req_dir)
    assert {(r["kind"], r["split"]) for r in recs} == {("main", "test"), ("screening", "train")}
    assert d2_io.resolve_split([env.tmp / "g" / "m1"], recs) == "test"
    assert d2_io.resolve_split([env.tmp / "g" / "m2"], recs) == "train"
    assert d2_io.resolve_split([env.tmp / "g" / "m2"], recs, "train") == "train"
    assert d2_io.resolve_split([env.tmp / "nowhere"], recs) == "test"
    assert d2_io.resolve_split([env.tmp / "nowhere"], recs, "train") == "train"
    with pytest.raises(d2_io.ScoreInputError, match="split must be"):
        d2_io.resolve_split([], recs, "dev")


def test_records_written_before_the_split_field_count_as_test_records(env):
    assert env.build("m1") == 0
    p = env.req_dir / "build_record.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    for k in ("split", "purpose"):
        rec.pop(k)
    p.write_text(json.dumps(rec), encoding="utf-8")
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    recs = d2_io.load_build_records(env.req_dir)
    assert d2_io.resolve_split([env.tmp / "g" / "m1"], recs) == "test"
    assert env.score(env.tmp / "g" / "m1") == 0
    assert d2_io.scored_metadata(env.tmp / "scored.parquet")["d2_split"] == "test"
    assert d2_io.scored_metadata(env.items_path) == {}


# ---------------------------------------------------------------- noise tranche on train items (scripts/precision_matched.py)

def _tranche_table(ids_by_model, models=("m1", "m2"), configs=("c1", "c2"), drop_last=False):
    rows = []
    for mi, mk in enumerate(models):
        for arm in ARMS:
            for ci, cid in enumerate(configs):
                for it in ids_by_model[mk]:
                    ok = (it + ci + mi + (arm == "zero_shot_cot")) % 3 != 0
                    rows.append({"model_key": mk, "arm": arm, "config_id": cid, "item_id": it, "strict_correct": int(ok),
                                 "lenient_correct": int(ok), "compliant": True, "finish_reason": "stop",
                                 "n_output_tokens": 10, "cached": False, "prompt_sha256": "0" * 64})
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype("int64")
    return df.iloc[:-1] if drop_last else df


@pytest.fixture
def pm_cli(monkeypatch):
    def fake_load(split="test", download_if_missing=False):
        n = 30 if split == "train" else 10
        return pd.DataFrame({"item_id": pd.Series(range(n), dtype="int64"), "question": ["q"] * n,
                             "gold_text": ["g"] * n, "gold_answer": ["1"] * n})
    monkeypatch.setattr(gsm8k, "load_gsm8k", fake_load)
    return load_script("precision_matched")


# ---------------------------------------------------------------- guards for the main analysis

def _clean_meta(sampler_sha="s1", items_sha="i1", scorer_sha=None):
    inp = {"model_key": "m1", "work_dir": "w", "test_backend": False, "request_kind": "main", "split": "test",
           "incomplete_allowed": False, "decoding_confirmed": True, "sampler_registered_sha256": sampler_sha}
    return {"d2_kind": "main", "d2_split": "test", "d2_scorer_status": "registered",
            "d2_inputs": json.dumps([inp]), "d2_items": json.dumps({"canonical_sha256": items_sha}),
            "d2_scorer_record": json.dumps({"manifest_sha256": scorer_sha})}


def test_main_problems_names_each_defect(tmp_path):
    md = _clean_meta()
    assert d2_io.main_problems(md, sampler={"sha256": "s1"}, items_info={"canonical_sha256": "i1"}) == []
    assert any("sampler" in p for p in d2_io.main_problems(md, sampler={"sha256": "other"}))
    assert any("pinned GSM8K test items" in p
               for p in d2_io.main_problems(md, items_info={"canonical_sha256": "i2"}))
    for field, value, words in (("test_backend", True, "test backend"), ("request_kind", None, "request kind"),
                                ("request_kind", "tranche", "request kind"), ("split", "train", "split"),
                                ("incomplete_allowed", True, "--allow-incomplete"),
                                ("incomplete_allowed", None, "--allow-incomplete"),
                                ("decoding_confirmed", False, "unconfirmed decoding")):
        bad = json.loads(md["d2_inputs"])
        bad[0][field] = value
        probs = d2_io.main_problems({**md, "d2_inputs": json.dumps(bad)})
        assert len(probs) == 1 and words in probs[0], (field, value, probs)
    assert "d2_inputs is absent" in d2_io.main_problems({k: v for k, v in md.items() if k != "d2_inputs"})[0]
    # the scorer manifest must be the one the table was scored with
    sc = tmp_path / "scorer.json"
    sc.write_text("{}")
    sha = hashlib.sha256(sc.read_bytes()).hexdigest()
    assert d2_io.main_problems(_clean_meta(scorer_sha=sha), scorer_path=sc) == []
    sc.write_text('{"changed": 1}')
    assert "scorer manifest" in d2_io.main_problems(_clean_meta(scorer_sha=sha), scorer_path=sc)[0]


def test_scored_echo_and_unmatched_tables_fail_the_main_guard(env, monkeypatch):
    # What score_d2 writes under --allow-test-backend / --allow-unmatched-requests never passes as a main table.
    assert env.build("m1") == 0
    env.generate("m1", env.tmp / "g" / "m1", backend="gold")
    scorer = _write_scorer(env.tmp / "gsm8k_scorer.json")
    monkeypatch.setattr(d2_io, "load_items", lambda p=None, manifest_path=None, split="test": (
        env.items, {"source": "fabricated", "canonical_sha256": "x", "n_items": N_ITEMS}))
    out = env.tmp / "reg.parquet"
    assert score_cli.main(["--work-dir", str(env.tmp / "g" / "m1"), "--scorer", str(scorer), "--out", str(out),
                           "--requests-dir", str(env.req_dir), "--allow-test-backend"]) == 0
    md = d2_io.scored_metadata(out)
    assert md["d2_kind"] == "main" and md["d2_scorer_status"] == "registered"
    probs = d2_io.main_problems(md, sampler=env.sampler, scorer_path=scorer)
    assert len(probs) == 1 and "test backend" in probs[0]
    out2 = env.tmp / "unmatched.parquet"
    assert score_cli.main(["--work-dir", str(env.tmp / "g" / "m1"), "--scorer", str(scorer), "--out", str(out2),
                           "--requests-dir", str(env.tmp / "elsewhere"), "--allow-test-backend",
                           "--allow-unmatched-requests", "--split", "test"]) == 0
    probs = d2_io.main_problems(d2_io.scored_metadata(out2), sampler=env.sampler)
    assert any("request kind None" in p for p in probs) and any("sampler" in p for p in probs)


def test_tranche_summary_keeps_accuracy_and_kinds_are_checked():
    table = pd.DataFrame({"model_key": ["m1"] * 2, "arm": ["direct_answer", "zero_shot_cot"], "config_id": ["c1"] * 2,
                          "item_id": [0, 0], "strict_correct": [1, 0], "lenient_correct": [1, 1],
                          "compliant": [True, True], "finish_reason": ["stop", "length"],
                          "n_output_tokens": [3, 9], "cached": [False, False]})
    tr = d2_io.score_summary(table, "tranche")
    assert tr[0]["strict_correct_pct"] == 100.0 and tr[1]["lenient_correct_pct"] == 100.0
    for kind in ("main", "screening"):
        assert not any("correct" in k for r in d2_io.score_summary(table, kind) for k in r)
    with pytest.raises(ValueError):
        d2_io.score_summary(table, "other")
    assert d2_io.scored_columns("screening") == d2_io.SCREENING_COLUMNS
    assert d2_io.scored_columns("tranche") == d2_io.SCORED_COLUMNS


def test_main_build_refuses_inadmissible_models_and_banks(env, capsys):
    env.write_admissibility(models=("m2",), excluded_models=("m1",))
    assert env.build("m1") == 2
    assert "excluded by the admissibility rules" in capsys.readouterr().err
    env.write_admissibility(models=("m2",))
    assert env.build("m1") == 2
    assert "never screened" in capsys.readouterr().err
    env.adm_path.write_text(json.dumps(admissibility_log(["c1"], excluded_configs=["c2"])))
    assert env.build("m1") == 2
    assert "excluded by the admissibility rules" in capsys.readouterr().err
    env.adm_path.unlink()
    assert env.build("m1") == 2
    assert "does not exist" in capsys.readouterr().err
    # tranche and screening builds come before the screening result and do not need the log
    assert env.build("m1", extra=["--configs", "c1", "--tranche"], sampler=False) == 0
    env.write_admissibility()
    assert env.build("m1") == 0
    assert env.record()["admissibility"]["sha256"] == hashlib.sha256(env.adm_path.read_bytes()).hexdigest()


def test_inspect_prepends_the_logged_prefill(env):
    assert env.build("m1") == 0
    g_dir = env.tmp / "g" / "m1"
    env.generate("m1", g_dir, backend="gold")
    plain, info = d2_io.inspect_work_dir(g_dir, allow_test_backend=True)
    assert info["prefill"] == {}
    log_path = sorted(g_dir.glob("run_log-*.json"))[0]
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log.setdefault("decoding", {})["prefill"] = {"direct_answer": "PRE:"}
    log_path.write_text(json.dumps(log), encoding="utf-8")
    out, info = d2_io.inspect_work_dir(g_dir, allow_test_backend=True)
    assert info["prefill"] == {"direct_answer": "PRE:"}
    da = (out["arm"] == "direct_answer").to_numpy()
    assert da.any() and (~da).any()
    assert (out.loc[da, "output_text"] == "PRE:" + plain.loc[da, "output_text"]).all()
    assert (out.loc[~da, "output_text"] == plain.loc[~da, "output_text"]).all()
    # two runs of one folder with different prefills are refused
    second = dict(log, run_id="second", decoding={**log["decoding"], "prefill": {}})
    (g_dir / "run_log-second.json").write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(d2_io.ScoreInputError, match="different prefills"):
        d2_io.inspect_work_dir(g_dir, allow_test_backend=True)


def test_scoring_refuses_a_folder_generated_on_two_gpu_models(env, capsys):
    assert env.build("m1") == 0
    g_dir = env.tmp / "g" / "m1"
    env.generate("m1", g_dir, backend="gold")
    log_path = sorted(g_dir.glob("run_log-*.json"))[0]
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log["gpu"] = {"count": 1, "devices": [{"name": "NVIDIA RTX A", "memory_mib": 1}]}
    log_path.write_text(json.dumps(log), encoding="utf-8")
    second = dict(log, run_id="second", gpu={"count": 1, "devices": [{"name": "NVIDIA RTX B", "memory_mib": 1}]})
    (g_dir / "run_log-second.json").write_text(json.dumps(second), encoding="utf-8")
    capsys.readouterr()
    assert env.score(g_dir) == 2
    assert "2 GPU models" in capsys.readouterr().err
    # one GPU model is recorded in the table metadata
    (g_dir / "run_log-second.json").unlink()
    out = env.tmp / "one.parquet"
    assert env.score(g_dir, out=out) == 0
    assert json.loads(d2_io.scored_metadata(out)["d2_inputs"])[0]["gpu_type"] == ["NVIDIA RTX A"]
