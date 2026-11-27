"""Tests for src/screening.py, scripts/screen_d2.py and config/admissibility.yaml.

Admissibility screening reports format and behaviour metrics on training-split calibration
items and never an outcome. The end-to-end tests run the real chain with the real
runner and a scripted test backend: scripts/build_requests.py --screening --split train, generation,
scripts/score_d2.py --screening, scripts/screen_d2.py. Nothing needs a GPU, the network or the repository
manifests: the calibration sample, the scorer manifest and the exclusion log all live in tmp_path.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_d2_io as h  # noqa: E402  (fabricated Env, fixtures and helpers shared with the d2_io tests)
from src import d2_io, generation as gen, scoring, screening  # noqa: E402

screen_cli = h.load_script("screen_d2")
ROOT = Path(__file__).resolve().parent.parent

N_SAMPLE = 6
CAP = 400                       # max_new_tokens of the test decoding config
MODELS = ("m1", "m2")
CONFIGS = ("c1", "c2", "c3")
ARMS = ("direct_answer", "zero_shot_cot")
OUTCOME = re.compile(r"correct|accura|gain", re.IGNORECASE)


# ---------------------------------------------------------------- scripted outputs

def make_text(kind, k):
    """(text, n_output_tokens, finish_reason) of one scripted output; k is the rank of the item in the sample."""
    return {
        "colon": (f"Answer: {k}", 4, "stop"),
        "colon_bold": (f"Let me reason.\n**Answer: {k}**", 12, "stop"),
        "isline": (f"Therefore the answer is {k}.", 6, "stop"),
        "hash": (f"Working it out.\n#### {k}", 8, "stop"),
        "empty": ("", 0, "stop"),
        "think": (f"<think>\nhmm\n</think>\nAnswer: {k}", 15, "stop"),
        "trunc": ("I keep thinking and thinking", CAP, "length"),
        "nonum": ("I cannot say.", 4, "stop"),
        "long": (f"Answer: {k}", 100, "stop"),
        "verylong": (f"Reasoning at length.\nAnswer: {k}", 200, "stop"),
    }[kind]


# (model, configuration, arm) -> the kind of output for each of the six sampled items
PLAN = {
    ("m1", "c1", "direct_answer"): ["colon"] * 6,
    ("m1", "c1", "zero_shot_cot"): ["colon_bold"] * 4 + ["trunc"] * 2,
    ("m1", "c2", "direct_answer"): ["isline"] * 5 + ["empty"],
    ("m1", "c2", "zero_shot_cot"): ["think"] * 2 + ["hash"] * 2 + ["nonum"] * 2,
    ("m1", "c3", "direct_answer"): ["colon"] * 3 + ["long"] * 3,
    ("m1", "c3", "zero_shot_cot"): ["colon"] * 6,
    ("m2", "c1", "direct_answer"): ["verylong"] * 6,
    ("m2", "c1", "zero_shot_cot"): ["colon"] * 6,
    ("m2", "c2", "direct_answer"): ["long"] * 6,
    ("m2", "c2", "zero_shot_cot"): ["trunc"] * 6,
    ("m2", "c3", "direct_answer"): ["verylong"] * 3 + ["colon"] * 3,
    ("m2", "c3", "zero_shot_cot"): ["colon_bold"] * 6,
}


class Scripted(gen.EchoBackend):
    """EchoBackend whose outputs follow PLAN; the configuration, arm and item are read from the prompt."""

    def __init__(self, spec, decoding, model_key, sample_ids):
        super().__init__(spec, decoding)
        self.model_key = model_key
        self.rank = {int(i): k for k, i in enumerate(sample_ids)}

    def generate(self, conversations, params):
        res = super().generate(conversations, params)
        out = []
        for conv, r in zip(conversations, res):
            user = conv[-1]["content"]
            cid = {"Q": "c1", "Problem": "c2", "Task": "c3"}[user.split(":", 1)[0]]
            arm = "zero_shot_cot" if "Think step by step" in user else "direct_answer"
            k = self.rank[int(re.search(r"Item (\d+):", user).group(1))]
            text, ntok, fin = make_text(PLAN[(self.model_key, cid, arm)][k], k)
            out.append(gen.GenResult(text, r.n_prompt_tokens, ntok, fin, r.prompt_echo))
        return out


def screening_bank():
    def cfg(prefix):
        return {"dimensions": {"prefix": prefix}, "arms": {
            "direct_answer": {"messages": [{"role": "user", "content": f"{prefix}: {{question}}\nGive the final number only."}]},
            "zero_shot_cot": {"messages": [{"role": "user", "content": f"{prefix}: {{question}}\nThink step by step."}]}}}
    return {"generator": "handwritten-test", "generator_version": "0", "seed": 0,
            "sampling_probabilities": {"c1": 0.25, "c2": 0.5, "c3": 0.25},
            "configurations": {"c1": cfg("Q"), "c2": cfg("Problem"), "c3": cfg("Task")}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "gpu_info", lambda timeout=10.0: {"count": 0, "devices": [], "driver_version": None,
                                                              "cuda_version_driver": None, "nvidia_smi": False})
    monkeypatch.setattr(screening, "SCORER_PATH", tmp_path / "no_scorer_manifest.json")   # never the repository's
    e = h.Env(tmp_path)
    e.bank = screening_bank()
    e.bank_path.write_text(json.dumps(e.bank), encoding="utf-8")
    e.items = pd.DataFrame({"item_id": pd.Series(range(12), dtype="int64"),
                            "question": [f"Item {i}: add {i} and {i}." for i in range(12)],
                            "gold_answer": [str(2 * i) for i in range(12)]})
    e.items.to_parquet(e.items_path, index=False)
    e.cal_path = tmp_path / "calibration_items.json"
    return e


def generate_scripted(env, key, sample_ids):
    spec = gen.parse_model_spec(key, gen.load_registry(env.models))
    be = Scripted(spec, gen.load_decoding(env.decoding), key, sample_ids)
    out_dir = env.tmp / "g" / key
    log = gen.run_generation(env.requests(key, sub="screening"), key, out_dir, backend="echo", backend_obj=be,
                             models_path=env.models, decoding_path=env.decoding, verbose=False, shard_size=1000)
    assert log["status"] == "complete"
    return out_dir


@pytest.fixture
def screened(env):
    """Requests built, generated and scored for both models: the input of the screening."""
    assert env.build_train(*MODELS, n=N_SAMPLE) == 0
    env.sample = json.loads(env.cal_path.read_text(encoding="utf-8"))["item_ids"]
    env.dirs = [generate_scripted(env, k, env.sample) for k in MODELS]
    env.scored = env.tmp / "scored" / "screening_scored.parquet"
    assert env.score(*env.dirs, out=env.scored,
                     extra=("--candidate-spec", "answer_colon", "--allow-test-backend", "--screening")) == 0
    env.out_dir = env.tmp / "report"
    env.log_path = env.tmp / "gsm8k_admissibility.json"
    env.scorer = h._write_scorer(env.tmp / "gsm8k_scorer.json", "answer_colon")
    return env


def write_rules(path, status="confirmed", **over):
    rules = {"status": status,
             "configurations": {
                 "max_compliance_shortfall_pp": {"direct_answer": 10, "zero_shot_cot": 10},
                 "max_truncation_pct": {"direct_answer": 10, "zero_shot_cot": 20},
                 "max_think_tag_pct": {"direct_answer": 0, "zero_shot_cot": 10}},
             "models": {"direct_answer_meaningful": {"length_tokens": 128, "max_share_above_pct": 25},
                        "thinking_off_holds": {"max_think_tag_pct": 10}}}
    for k, v in over.items():  # a models override replaces only the rules it names
        rules[k] = {**rules[k], **v} if k == "models" else v
    if status == "PENDING_REVIEW":
        rules["configurations"] = {k: {a: None for a in ARMS} for k in rules["configurations"]}
        rules["models"] = {"direct_answer_meaningful": {"length_tokens": None, "max_share_above_pct": None},
                           "thinking_off_holds": {"max_think_tag_pct": None}}
    path.write_text(yaml.safe_dump(rules), encoding="utf-8")
    return path


def run_screen(env, rules, *extra, scorer=True):
    args = ["--scored", str(env.scored), "--rules", str(rules), "--out-dir", str(env.out_dir),
            "--exclusion-log", str(env.log_path), *extra]
    if scorer:
        args += ["--scorer", str(env.scorer)]
    return screen_cli.main(args)


def report_of(env):
    return json.loads((env.out_dir / "screening_report.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- independent expectations from PLAN

def pct(k, n):
    return 100.0 * k / n


def cell_rows(models, cid, arm):
    return [make_text(kind, k) for m in models for k, kind in enumerate(PLAN[(m, cid, arm)])]


def expected(models, cid, arm, spec=None):
    rows = cell_rows(models, cid, arm)
    texts, toks, fins = [r[0] for r in rows], np.array([r[1] for r in rows], float), [r[2] for r in rows]
    n = len(rows)
    out = {"n": n,
           "lenient": pct(sum(scoring.lenient_extract(t) is not None for t in texts), n),
           "truncated": pct(sum(f == "length" for f in fins), n),
           "think": pct(sum("<think>" in t for t in texts), n),
           "empty": pct(sum(not t.strip() for t in texts), n),
           "mean": float(toks.mean()), "median": float(np.median(toks)), "p95": float(np.percentile(toks, 95)),
           "above": {L: pct(int((toks > L).sum()), n) for L in (32, 64, 128)}}
    if spec is not None:
        out["strict"] = pct(sum(scoring.strict_extract(t, spec) is not None for t in texts), n)
    return out


# ---------------------------------------------------------------- the rules file

def test_repository_rules_file_is_valid_and_pending_means_all_null():
    rules, prov = screening.load_rules()
    assert prov["path"] == "config/admissibility.yaml" and len(prov["sha256"]) == 64
    assert screening.validate_rules(rules) == []
    if rules["status"] == screening.STATUS_PENDING:
        flat = [v for k in rules["configurations"].values() for v in k.values()]
        flat += [v for k in rules["models"].values() for v in k.values()]
        assert flat and all(v is None for v in flat)
    text = (ROOT / "config" / "admissibility.yaml").read_text(encoding="utf-8")
    assert "PENDING_REVIEW" in text


def test_the_confirmed_compliance_rule_is_relative_to_the_median_configuration():
    rules, _ = screening.load_rules()
    assert rules["status"] == screening.STATUS_CONFIRMED          # the rules file is confirmed
    assert rules["configurations"]["max_compliance_shortfall_pp"] == {"direct_answer": 10, "zero_shot_cot": 10}
    old = {**rules, "configurations": {**rules["configurations"],
                                       "min_strict_compliance_pct": {"direct_answer": 80, "zero_shot_cot": 80}}}
    assert any("unknown rule 'min_strict_compliance_pct'" in p for p in screening.validate_rules(old))


def _rules(**kw):
    r = yaml.safe_load(write_rules(Path(kw.pop("tmp")) / "r.yaml").read_text())
    return r


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.update(status="draft"), "status must be"),
    (lambda r: r.update(extra=1), "unknown top-level keys"),
    (lambda r: r["configurations"].update(max_wrong_pct={"direct_answer": 1, "zero_shot_cot": 1}), "unknown rule"),
    (lambda r: r["configurations"]["max_truncation_pct"].pop("zero_shot_cot"), "exactly the arms"),
    (lambda r: r["configurations"]["max_truncation_pct"].update(direct_answer=101), "between 0 and 100"),
    (lambda r: r["configurations"]["max_truncation_pct"].update(direct_answer=-1), "between 0 and 100"),
    (lambda r: r["configurations"]["max_truncation_pct"].update(direct_answer="5"), "between 0 and 100"),
    (lambda r: r["configurations"]["max_truncation_pct"].update(direct_answer=True), "between 0 and 100"),
    (lambda r: r["configurations"].pop("max_think_tag_pct"), "max_think_tag_pct must be a mapping"),
    (lambda r: r["models"]["direct_answer_meaningful"].update(length_tokens=50), "must be one of"),
    (lambda r: r["models"]["direct_answer_meaningful"].update(max_share_above_pct=120), "between 0 and 100"),
    (lambda r: r["models"].update(other=1), "exactly the rule"),
    (lambda r: r["models"]["direct_answer_meaningful"].pop("length_tokens"), "exactly length_tokens"),
    (lambda r: r["models"].pop("thinking_off_holds"), "exactly the rules"),
    (lambda r: r["models"]["thinking_off_holds"].update(other=1), "exactly max_think_tag_pct"),
    (lambda r: r["models"]["thinking_off_holds"].update(max_think_tag_pct=120), "between 0 and 100"),
    (lambda r: r["models"]["thinking_off_holds"].update(max_think_tag_pct=None), "thresholds are null"),
    (lambda r: r["configurations"]["max_truncation_pct"].update(direct_answer=None), "thresholds are null"),
])
def test_rules_validation(tmp_path, mutate, match):
    r = _rules(tmp=tmp_path)
    mutate(r)
    problems = screening.validate_rules(r)
    assert any(match in p for p in problems), problems


def test_pending_rules_may_hold_nulls_and_confirmed_rules_may_not(tmp_path):
    assert screening.validate_rules(_rules(tmp=tmp_path)) == []
    pending = yaml.safe_load(write_rules(tmp_path / "p.yaml", "PENDING_REVIEW").read_text())
    assert screening.validate_rules(pending) == []
    pending["status"] = "confirmed"
    problems = screening.validate_rules(pending)
    assert len(problems) == 1 and "thresholds are null" in problems[0] and "trivial bound" in problems[0]
    assert screening.validate_rules([]) == ["the rules file must be a mapping"]
    with pytest.raises(screening.ScreeningError, match="cannot read"):
        screening.load_rules(tmp_path / "absent.yaml")
    (tmp_path / "bad.yaml").write_text("status: [", encoding="utf-8")
    with pytest.raises(screening.ScreeningError, match="not valid YAML"):
        screening.load_rules(tmp_path / "bad.yaml")


# ---------------------------------------------------------------- metrics on hand-made rows

def _frame(texts, tokens, finishes=None):
    n = len(texts)
    return pd.DataFrame({"model_key": "m", "arm": "direct_answer", "config_id": "c", "item_id": range(n),
                         "n_output_tokens": tokens, "finish_reason": finishes or ["stop"] * n, "output_text": texts})


def test_row_metrics_flags_each_behaviour():
    specs = {"colon": scoring.CANDIDATE_SPECS["answer_colon"], "hash": scoring.CANDIDATE_SPECS["hash_n"]}
    df = _frame(["Answer: 5", "work\n#### 5", "", "   \n\t", "<THINK>x</Think>\nAnswer: 7", "no digits here", None, "x </think> 5"],
                [3, 4, 0, 1, 9, 5, 0, 6], ["stop", "stop", "stop", "stop", "stop", "stop", "stop", "length"])
    r = screening.row_metrics(df, specs, scoring.CANDIDATE_SPECS["answer_colon"])
    assert list(r["cand__colon"]) == [True, False, False, False, True, False, False, False]
    assert list(r["cand__hash"]) == [False, True, False, False, False, False, False, False]
    assert list(r["registered"]) == list(r["cand__colon"])
    assert list(r["lenient"]) == [True, True, False, False, True, False, False, True]
    assert list(r["empty"]) == [False, False, True, True, False, False, True, False]
    assert list(r["think_tag"]) == [False, False, False, False, True, False, False, True]
    assert list(r["truncated"]) == [False] * 7 + [True]
    assert "registered" not in screening.row_metrics(df, specs).columns
    assert not [c for c in r.columns if OUTCOME.search(c)]


def test_summarise_numbers_and_the_direct_diagnostic_only_in_the_direct_arm():
    df = _frame(["Answer: 1"] * 5, [1, 2, 3, 4, 100], ["stop"] * 4 + ["length"])
    specs = {"colon": scoring.CANDIDATE_SPECS["answer_colon"]}
    rows = screening.row_metrics(df, specs)
    s = screening.summarise(rows, "direct_answer", ["colon"], False)
    assert s["n"] == 5 and s["strict_compliance_pct"] == {"colon": 100.0} and s["registered_compliance_pct"] is None
    assert s["truncated_pct"] == 20.0 and s["empty_pct"] == 0.0 and s["think_tag_pct"] == 0.0
    assert s["lenient_extractable_pct"] == 100.0
    assert s["output_tokens_mean"] == 22.0 and s["output_tokens_median"] == 3.0
    assert s["output_tokens_p95"] == pytest.approx(80.8)             # numpy linear interpolation: 4 + 0.8 * (100 - 4)
    assert s["direct_answer_diagnostic"] == {"above_32_tokens_pct": 20.0, "above_64_tokens_pct": 20.0,
                                             "above_128_tokens_pct": 0.0}
    assert screening.summarise(rows, "zero_shot_cot", ["colon"], False)["direct_answer_diagnostic"] is None
    assert screening.DIRECT_LENGTHS == (32, 64, 128)


def test_outcome_guard_rejects_correctness_accuracy_and_gain_keys():
    screening.assert_no_outcome_keys({"a": [{"b": 1}], "strict_compliance_pct": {"x": 1}})
    for bad in ({"strict_correct": 1}, {"x": {"lenient_correct": 1}}, {"x": [{"accuracy_pct": 1}]}, {"gain": 1},
                {"Mean_Accuracy": 1}):
        with pytest.raises(screening.ScreeningError, match="names an outcome"):
            screening.assert_no_outcome_keys(bad)
    with pytest.raises(screening.ScreeningError):
        screening.assert_no_outcome_keys(pd.DataFrame({"strict_correct": [1]}))
    assert "correct" not in " ".join(screening.SCORED_READ_COLUMNS)


# ---------------------------------------------------------------- end to end: the metrics report (rules pending)

def test_metrics_report_covers_every_metric_per_cell_configuration_and_model(screened, capsys):
    rules = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    capsys.readouterr()
    assert run_screen(screened, rules, scorer=False) == 0
    cap = capsys.readouterr()
    assert "PENDING_REVIEW" in cap.err
    res = json.loads(cap.out)
    assert res["rules_applied"] is False and res["exclusion_log"] is None and res["registered_scorer"] == "absent"
    assert not screened.log_path.exists()                                  # no manifest while pending
    rep = report_of(screened)
    assert rep["inputs"]["split"] == "train" and rep["inputs"]["models"] == list(MODELS)
    assert rep["inputs"]["n_items"] == N_SAMPLE and rep["inputs"]["item_set_identical_across_cells"] is True
    assert rep["inputs"]["n_rows"] == 2 * 3 * 2 * N_SAMPLE and rep["inputs"]["test_backend"] is True
    assert rep["inputs"]["calibration_items"]["sha256"] == json.loads(screened.cal_path.read_text())["sha256"]
    assert rep["rules"]["applied"] is False and rep["exclusions"] is None and "PENDING_REVIEW" in rep["rules"]["note"]
    assert rep["scorers"]["registered"]["status"] == "absent"
    # every candidate spec is read at run time
    names = list(scoring.CANDIDATE_SPECS)
    assert rep["scorers"]["candidate_specs"] == names and len(names) >= 4

    assert len(rep["cells"]) == 2 * 3 * 2 and len(rep["by_model"]) == 2 * 2 and len(rep["by_configuration"]) == 3 * 2
    for c in rep["cells"]:
        e = expected([c["model_key"]], c["config_id"], c["arm"])
        assert c["n"] == e["n"] == N_SAMPLE
        assert list(c["strict_compliance_pct"]) == names and c["registered_compliance_pct"] is None
        for name, spec in scoring.CANDIDATE_SPECS.items():
            assert c["strict_compliance_pct"][name] == pytest.approx(expected([c["model_key"]], c["config_id"], c["arm"], spec)["strict"])
        assert c["lenient_extractable_pct"] == pytest.approx(e["lenient"])
        assert c["truncated_pct"] == pytest.approx(e["truncated"]) and c["think_tag_pct"] == pytest.approx(e["think"])
        assert c["empty_pct"] == pytest.approx(e["empty"])
        assert (c["output_tokens_mean"], c["output_tokens_median"], c["output_tokens_p95"]) == pytest.approx(
            (e["mean"], e["median"], e["p95"]))
        if c["arm"] == "direct_answer":
            assert c["direct_answer_diagnostic"] == pytest.approx({f"above_{L}_tokens_pct": v for L, v in e["above"].items()})
        else:
            assert c["direct_answer_diagnostic"] is None
    # aggregated per configuration (pooled over models) and per model (pooled over configurations)
    for c in rep["by_configuration"]:
        e = expected(MODELS, c["config_id"], c["arm"])
        assert c["n"] == 2 * N_SAMPLE and c["truncated_pct"] == pytest.approx(e["truncated"])
        assert c["think_tag_pct"] == pytest.approx(e["think"]) and c["empty_pct"] == pytest.approx(e["empty"])
        assert c["output_tokens_p95"] == pytest.approx(e["p95"]) and c["output_tokens_median"] == pytest.approx(e["median"])
        assert "model_key" not in c
    for m in rep["by_model"]:
        rows = [r for cid in CONFIGS for r in cell_rows([m["model_key"]], cid, m["arm"])]
        toks = np.array([r[1] for r in rows], float)
        assert m["n"] == 3 * N_SAMPLE and m["output_tokens_mean"] == pytest.approx(toks.mean())
        assert m["truncated_pct"] == pytest.approx(pct(sum(r[2] == "length" for r in rows), len(rows)))
        if m["arm"] == "direct_answer":
            assert m["direct_answer_diagnostic"]["above_128_tokens_pct"] == pytest.approx(pct(int((toks > 128).sum()), len(toks)))
    by_model = {(m["model_key"], m["arm"]): m for m in rep["by_model"]}
    assert by_model[("m2", "direct_answer")]["direct_answer_diagnostic"]["above_128_tokens_pct"] == pytest.approx(50.0)
    assert by_model[("m1", "direct_answer")]["direct_answer_diagnostic"]["above_32_tokens_pct"] == pytest.approx(pct(3, 18))
    # specific known cells
    cells = {(c["model_key"], c["config_id"], c["arm"]): c for c in rep["cells"]}
    assert cells[("m1", "c2", "direct_answer")]["empty_pct"] == pytest.approx(pct(1, 6))
    assert cells[("m1", "c1", "zero_shot_cot")]["truncated_pct"] == pytest.approx(pct(2, 6))
    assert cells[("m1", "c2", "zero_shot_cot")]["think_tag_pct"] == pytest.approx(pct(2, 6))
    # markdown copies
    md = (screened.out_dir / "screening_report.md").read_text(encoding="utf-8")
    clean = (screened.out_dir / "screening_report_clean.md").read_text(encoding="utf-8")
    assert md == clean and "By model and arm" in md and "By configuration and arm" in md
    assert all(f"{n} %" in md for n in names) and "c3" in md and "PENDING_REVIEW" in md


def test_registered_compliance_is_reported_when_the_scorer_manifest_exists(screened):
    rules = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    assert run_screen(screened, rules) == 0                                # --scorer given
    rep = report_of(screened)
    spec = scoring.CANDIDATE_SPECS["answer_colon"]
    assert rep["scorers"]["registered"]["status"] == "registered" and rep["scorers"]["registered"]["format_spec"]["name"] == "answer_colon"
    for c in rep["cells"]:
        assert c["registered_compliance_pct"] == pytest.approx(expected([c["model_key"]], c["config_id"], c["arm"], spec)["strict"])
        assert c["registered_compliance_pct"] == pytest.approx(c["strict_compliance_pct"]["answer_colon"])
    # the default path is used when the manifest exists there
    shutil_target = screened.tmp / "default_scorer.json"
    shutil_target.write_bytes(screened.scorer.read_bytes())
    screening.SCORER_PATH = shutil_target
    try:
        assert run_screen(screened, rules, scorer=False) == 0
    finally:
        screening.SCORER_PATH = screened.tmp / "no_scorer_manifest.json"
    assert report_of(screened)["scorers"]["registered"]["status"] == "registered"
    # an explicit path that does not exist is refused
    assert run_screen(screened, rules, "--scorer", str(screened.tmp / "absent.json"), scorer=False) == 2


def test_screening_computes_no_correctness_anywhere(screened, monkeypatch):
    """No correctness, accuracy or gain: not in any key or column, not read from the scored table, never scored."""
    def forbidden(*a, **k):
        raise AssertionError("screening must not call score_outputs (it would compute correctness)")
    monkeypatch.setattr(scoring, "score_outputs", forbidden)
    reads = []
    real = pd.read_parquet

    def spy(path, *a, **kw):
        reads.append((str(path), kw.get("columns")))
        return real(path, *a, **kw)
    monkeypatch.setattr(pd, "read_parquet", spy)
    rules = write_rules(screened.tmp / "conf.yaml")
    assert run_screen(screened, rules) == 0

    scored_reads = [c for p, c in reads if p.endswith("screening_scored.parquet")]
    assert scored_reads == [screening.SCORED_READ_COLUMNS]                  # the correctness columns are never loaded
    rep = report_of(screened)
    screening.assert_no_outcome_keys(rep)
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rep)
    walk(json.loads(screened.log_path.read_text(encoding="utf-8")))
    assert keys and not [k for k in keys if OUTCOME.search(str(k))]
    text = (screened.out_dir / "screening_report.json").read_text(encoding="utf-8")
    log_text = screened.log_path.read_text(encoding="utf-8")
    for blob in (text, log_text, (screened.out_dir / "screening_report.md").read_text(encoding="utf-8")):
        assert "strict_correct" not in blob and "lenient_correct" not in blob
    frame = screening.row_metrics(pd.DataFrame({
        "model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": [0], "n_output_tokens": [1],
        "finish_reason": ["stop"], "output_text": ["Answer: 1"]}), dict(scoring.CANDIDATE_SPECS))
    assert not [c for c in frame.columns if OUTCOME.search(c)]


def test_work_dirs_default_to_the_scored_tables_metadata_and_can_be_given(screened):
    rules = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    md = d2_io.scored_metadata(screened.scored)
    assert screening.work_dirs_from_metadata(md) == [str(d) for d in screened.dirs]
    assert run_screen(screened, rules, "--work-dir", *map(str, screened.dirs)) == 0
    rep_given = report_of(screened)
    assert run_screen(screened, rules) == 0
    rep_default = report_of(screened)
    assert rep_given["cells"] == rep_default["cells"] and rep_given["inputs"]["work_dirs"] == rep_default["inputs"]["work_dirs"]


# ---------------------------------------------------------------- rules applied (status confirmed)

def test_confirmed_rules_write_the_exclusion_log(screened, capsys):
    rules = write_rules(screened.tmp / "conf.yaml")
    capsys.readouterr()
    assert run_screen(screened, rules) == 0
    cap = capsys.readouterr()
    assert "PENDING_REVIEW" not in cap.err
    res = json.loads(cap.out)
    assert res["rules_applied"] is True and res["exclusion_log"] == str(screened.log_path)
    assert res["excluded_models"] == ["m2"] and res["excluded_configurations"] == ["c1", "c2"]
    log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    assert log["kind"] == "admissibility_exclusion_log" and log["split"] == "train"
    assert log["retained_models"] == ["m1"] and log["retained_configuration_ids"] == ["c3"]
    assert log["configurations_screened"] == list(CONFIGS) and log["models_screened"] == list(MODELS)
    # models: m2 reasons in the direct arm (9 of 18 outputs longer than 128 tokens against a 25 percent maximum)
    (m,) = log["excluded_models"]
    assert m["model_key"] == "m2" and m["rule"] == "direct_answer_meaningful"
    assert m["metric"] == "direct_answer_above_128_tokens_pct" and m["value"] == pytest.approx(50.0)
    assert m["threshold"] == 25.0 and m["comparison"] == "value > threshold" and m["n"] == 18
    # configurations: pooled over the retained model m1 only
    ex = {c["config_id"]: c["violations"] for c in log["excluded_configurations"]}
    assert set(ex) == {"c1", "c2"}
    got = {(c, v["arm"], v["rule"]): v for c, vs in ex.items() for v in vs}
    assert set(got) == {("c1", "zero_shot_cot", "max_compliance_shortfall_pp"), ("c1", "zero_shot_cot", "max_truncation_pct"),
                        ("c2", "direct_answer", "max_compliance_shortfall_pp"), ("c2", "zero_shot_cot", "max_think_tag_pct")}
    # pooled over m1: direct 100 / 0 / 100 (median 100), chain of thought 0 / 33.3 / 100 (median 33.3)
    v = got[("c1", "zero_shot_cot", "max_compliance_shortfall_pp")]
    assert v["metric"] == "registered_compliance_shortfall_pp" and v["value"] == pytest.approx(pct(2, 6))
    assert v["compliance_pct"] == 0.0 and v["arm_median_compliance_pct"] == pytest.approx(pct(2, 6))
    assert v["threshold"] == 10.0 and v["comparison"] == "value > threshold" and v["n"] == N_SAMPLE
    assert v["pooled_over_models"] == ["m1"]
    assert got[("c1", "zero_shot_cot", "max_truncation_pct")]["value"] == pytest.approx(pct(2, 6))
    assert got[("c1", "zero_shot_cot", "max_truncation_pct")]["metric"] == "truncated_pct"
    d = got[("c2", "direct_answer", "max_compliance_shortfall_pp")]
    assert d["value"] == 100.0 and d["compliance_pct"] == 0.0 and d["arm_median_compliance_pct"] == 100.0
    assert got[("c2", "zero_shot_cot", "max_think_tag_pct")]["value"] == pytest.approx(pct(2, 6))
    assert got[("c2", "zero_shot_cot", "max_think_tag_pct")]["metric"] == "think_tag_pct"
    # provenance
    bank_sha = d2_io.load_bank(screened.bank_path)[1]
    assert log["inputs"]["bank_sha256"] == bank_sha and log["rules"]["status"] == "confirmed" and log["rules"]["applied"] is True
    assert log["inputs"]["calibration_items"]["sha256"] == json.loads(screened.cal_path.read_text())["sha256"]
    assert log["inputs"]["scored_sha256"] == hashlib.sha256(screened.scored.read_bytes()).hexdigest()
    assert log["scorer"]["status"] == "registered" and log["scorer"]["scorer_version"] == scoring.SCORER_VERSION
    assert log["report"]["sha256"] == hashlib.sha256((screened.out_dir / "screening_report.json").read_bytes()).hexdigest()
    assert log["rules"]["thresholds"]["models"]["direct_answer_meaningful"]["max_share_above_pct"] == 25
    rep = report_of(screened)
    assert rep["exclusions"]["retained_configuration_ids"] == ["c3"]
    assert "Excluded configurations: c1, c2" in (screened.out_dir / "screening_report.md").read_text(encoding="utf-8")


def test_a_value_equal_to_a_threshold_passes(screened):
    """Maximum share 50 keeps m2 (exactly 50 percent above 128 tokens). Pooled over both models, direct compliance is
    100 / 50 / 100 (median 100), so c2 falls short by exactly the 50-point maximum set for that arm and passes;
    chain of thought is 50 / 16.7 / 50 (median 50), so c2 falls short by 33.3 points. Only c2 fails (compliance,
    truncation, think tags), in the chain-of-thought arm."""
    rules = write_rules(screened.tmp / "boundary.yaml", models={"direct_answer_meaningful": {
        "length_tokens": 128, "max_share_above_pct": 50}}, configurations={
        "max_compliance_shortfall_pp": {"direct_answer": 50, "zero_shot_cot": 10},
        "max_truncation_pct": {"direct_answer": 10, "zero_shot_cot": 20},
        "max_think_tag_pct": {"direct_answer": 0, "zero_shot_cot": 10}})
    assert run_screen(screened, rules) == 0
    log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    assert log["excluded_models"] == [] and log["retained_models"] == ["m1", "m2"]
    assert [c["config_id"] for c in log["excluded_configurations"]] == ["c2"]
    assert log["retained_configuration_ids"] == ["c1", "c3"]
    vs = {(v["arm"], v["rule"]): v for v in log["excluded_configurations"][0]["violations"]}
    assert set(vs) == {("zero_shot_cot", "max_compliance_shortfall_pp"), ("zero_shot_cot", "max_truncation_pct"),
                       ("zero_shot_cot", "max_think_tag_pct")}
    assert vs[("zero_shot_cot", "max_compliance_shortfall_pp")]["value"] == pytest.approx(50.0 - pct(2, 12))
    assert vs[("zero_shot_cot", "max_compliance_shortfall_pp")]["compliance_pct"] == pytest.approx(pct(2, 12))
    assert vs[("zero_shot_cot", "max_truncation_pct")]["value"] == pytest.approx(50.0)
    assert vs[("zero_shot_cot", "max_think_tag_pct")]["value"] == pytest.approx(pct(2, 12))
    assert vs[("zero_shot_cot", "max_think_tag_pct")]["pooled_over_models"] == ["m1", "m2"]
    assert vs[("zero_shot_cot", "max_think_tag_pct")]["n"] == 12


def test_a_model_whose_outputs_carry_think_tags_is_excluded(screened):
    """m1 writes a think tag in 2 of its 36 outputs (c2, chain of thought). A maximum of 5 percent excludes it, the
    exact share keeps it; m2 writes none."""
    lenient = {"direct_answer_meaningful": {"length_tokens": 128, "max_share_above_pct": 50}}
    rules = write_rules(screened.tmp / "think.yaml", models={**lenient, "thinking_off_holds": {"max_think_tag_pct": 5}})
    assert run_screen(screened, rules) == 0
    log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    (m,) = log["excluded_models"]
    assert m["model_key"] == "m1" and m["rule"] == "thinking_off_holds" and m["metric"] == "think_tag_pct_all_outputs"
    assert m["value"] == pytest.approx(pct(2, 36)) and m["threshold"] == 5.0 and m["n"] == 36
    assert m["comparison"] == "value > threshold" and log["retained_models"] == ["m2"]
    assert all(v["pooled_over_models"] == ["m2"] for c in log["excluded_configurations"] for v in c["violations"])
    exact = write_rules(screened.tmp / "exact.yaml",
                        models={**lenient, "thinking_off_holds": {"max_think_tag_pct": pct(2, 36)}})
    assert run_screen(screened, exact, "--overwrite") == 0
    assert json.loads(screened.log_path.read_text(encoding="utf-8"))["retained_models"] == ["m1", "m2"]


def test_when_every_model_is_excluded_no_configuration_is_assessed(screened):
    rules = write_rules(screened.tmp / "none.yaml", models={"direct_answer_meaningful": {
        "length_tokens": 32, "max_share_above_pct": 0}})
    assert run_screen(screened, rules) == 0
    log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    assert {m["model_key"] for m in log["excluded_models"]} == {"m1", "m2"} and log["retained_models"] == []
    assert log["configuration_rules_evaluated"] is False and "no model was retained" in log["configuration_rules_note"]
    assert log["excluded_configurations"] == [] and log["retained_configuration_ids"] == []
    assert next(m for m in log["excluded_models"] if m["model_key"] == "m1")["value"] == pytest.approx(pct(3, 18))


def test_trivial_bounds_exclude_nothing(screened):
    rules = write_rules(screened.tmp / "open.yaml",
                        configurations={"max_compliance_shortfall_pp": {a: 100 for a in ARMS},
                                        "max_truncation_pct": {a: 100 for a in ARMS},
                                        "max_think_tag_pct": {a: 100 for a in ARMS}},
                        models={"direct_answer_meaningful": {"length_tokens": 64, "max_share_above_pct": 100},
                                "thinking_off_holds": {"max_think_tag_pct": 100}})
    assert run_screen(screened, rules) == 0
    log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    assert log["excluded_models"] == [] and log["excluded_configurations"] == []
    assert log["retained_configuration_ids"] == list(CONFIGS) and log["retained_models"] == list(MODELS)


def test_exclusion_log_is_stable_and_not_overwritten_silently(screened):
    rules = write_rules(screened.tmp / "conf.yaml")
    assert run_screen(screened, rules) == 0
    first_log = json.loads(screened.log_path.read_text(encoding="utf-8"))
    report_bytes = (screened.out_dir / "screening_report.json").read_bytes()
    log_bytes = screened.log_path.read_bytes()
    assert run_screen(screened, rules) == 0                                # identical content: accepted again
    again = json.loads(screened.log_path.read_text(encoding="utf-8"))
    assert {k: v for k, v in again.items() if k not in ("created", "report")} == {
        k: v for k, v in first_log.items() if k not in ("created", "report")}
    assert again["report"]["sha256"] == hashlib.sha256((screened.out_dir / "screening_report.json").read_bytes()).hexdigest()
    report_bytes = (screened.out_dir / "screening_report.json").read_bytes()
    log_bytes = screened.log_path.read_bytes()
    other = write_rules(screened.tmp / "other.yaml", models={"direct_answer_meaningful": {
        "length_tokens": 128, "max_share_above_pct": 50}})
    assert run_screen(screened, other) == 2                                # other content: refused before anything is written
    assert screened.log_path.read_bytes() == log_bytes
    assert (screened.out_dir / "screening_report.json").read_bytes() == report_bytes
    assert run_screen(screened, other, "--overwrite") == 0
    assert json.loads(screened.log_path.read_text())["retained_models"] == ["m1", "m2"]


def test_pending_rules_never_write_a_manifest_even_when_one_exists_elsewhere(screened):
    rules = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    assert run_screen(screened, rules) == 0
    assert not screened.log_path.exists()
    assert sorted(p.name for p in screened.out_dir.iterdir()) == [
        "screening_report.json", "screening_report.md", "screening_report_clean.md"]


# ---------------------------------------------------------------- refusals

def test_confirmed_rules_need_the_registered_scorer(screened, capsys):
    rules = write_rules(screened.tmp / "conf.yaml")
    capsys.readouterr()
    assert run_screen(screened, rules, scorer=False) == 2                   # SCORER_PATH points at an absent file
    assert "scorer manifest" in capsys.readouterr().err
    assert not screened.log_path.exists() and not screened.out_dir.exists()


def test_invalid_or_missing_rules_and_scored_inputs_are_refused(screened, capsys):
    bad = screened.tmp / "bad.yaml"
    bad.write_text(yaml.safe_dump({"status": "confirmed"}), encoding="utf-8")
    capsys.readouterr()
    assert run_screen(screened, bad) == 2 and "invalid" in capsys.readouterr().err
    assert run_screen(screened, screened.tmp / "absent.yaml") == 2
    pending = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    assert screen_cli.main(["--scored", str(screened.tmp / "absent.parquet"), "--rules", str(pending),
                            "--out-dir", str(screened.out_dir)]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert not screened.out_dir.exists()


def test_only_train_screening_tables_are_accepted(screened, capsys):
    pending = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    df = pd.read_parquet(screened.scored)
    md = d2_io.scored_metadata(screened.scored)
    capsys.readouterr()
    for kind, split, msg in (("tranche", "train", "not a screening table"), ("main", "test", "not a screening table"),
                             ("screening", "test", "training items only")):
        p = d2_io.write_scored(df, screened.tmp / f"{kind}_{split}.parquet", {**md, "d2_kind": kind, "d2_split": split})
        assert screen_cli.main(["--scored", str(p), "--rules", str(pending), "--out-dir", str(screened.out_dir)]) == 2
        assert msg in capsys.readouterr().err
    p = d2_io.write_scored(df, screened.tmp / "nometa.parquet", {})
    assert screen_cli.main(["--scored", str(p), "--rules", str(pending), "--out-dir", str(screened.out_dir)]) == 2
    assert not screened.out_dir.exists()


def test_generation_folders_must_be_the_ones_that_were_scored(screened, capsys):
    pending = write_rules(screened.tmp / "pending.yaml", "PENDING_REVIEW")
    capsys.readouterr()
    assert run_screen(screened, pending, "--work-dir", str(screened.dirs[0])) == 2           # m2 outputs missing
    assert "no output in the generation folders" in capsys.readouterr().err
    assert run_screen(screened, pending, "--work-dir", str(screened.tmp / "missing")) == 2
    assert "does not exist" in capsys.readouterr().err
    df = pd.read_parquet(screened.scored)
    df.loc[0, "n_output_tokens"] += 1                                                        # the table no longer matches
    md = d2_io.scored_metadata(screened.scored)
    tampered = d2_io.write_scored(df, screened.tmp / "tampered.parquet", md)
    assert screen_cli.main(["--scored", str(tampered), "--rules", str(pending), "--out-dir", str(screened.out_dir)]) == 2
    assert "disagree with the scored table on n_output_tokens" in capsys.readouterr().err
    assert not screened.out_dir.exists()
    # a scored table without folders in its metadata needs --work-dir
    md2 = {k: v for k, v in md.items() if k != "d2_inputs"}
    nodirs = d2_io.write_scored(pd.read_parquet(screened.scored), screened.tmp / "nodirs.parquet", md2)
    assert screen_cli.main(["--scored", str(nodirs), "--rules", str(pending), "--out-dir", str(screened.out_dir)]) == 2
    assert "pass --work-dir" in capsys.readouterr().err
    assert screen_cli.main(["--scored", str(nodirs), "--rules", str(pending), "--out-dir", str(screened.out_dir),
                            "--work-dir", *map(str, screened.dirs)]) == 0
