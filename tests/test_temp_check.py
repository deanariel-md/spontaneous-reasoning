"""Tests of the temperature 0.7 spot check: the runner's opt-in sampling mode
(src/generation.py, Decoding.sampling), the decoding files, the model rule, the request files, the in-job gate, the
analysis and the macros. No network, no GPU."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import generation as g

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper"))
import temp_check as tc  # noqa: E402
import temp_check_gate as gate  # noqa: E402
import macros_temp_check as mtc  # noqa: E402

SHA = "a" * 40
KEY = "m1"
ARMS3 = ("direct_answer", "forced_direct", "zero_shot_cot")


# ---------------------------------------------------------------- helpers

def entry():
    return {"family": "test", "hf_id": "org/model", "revision": SHA, "dtype": "bfloat16", "quantization": "none",
            "tokenizer_mode": "auto", "max_model_len": 8192, "tensor_parallel_size": 1,
            "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
            "thinking_off": {"method": "none_needed", "chat_template_kwargs": {}, "system_prompt_token": None,
                             "rationale": "test"}, "kernels": "batch_invariant"}


def decoding(**over):
    d = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
         "max_new_tokens": {"direct_answer": 400, "zero_shot_cot": 400, "forced_direct": 16},
         "prefill": {"forced_direct": "\\boxed{"}, "stop": [], "include_stop_str_in_output": False,
         "skip_special_tokens": True,
         "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                    "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}
    d.update(over)
    return d


SAMPLED = {"temperature": 0.7, "sampling": {"seed": "request"}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 1, "devices": [{"name": "Fake Ada", "memory_mib": 49140}], "driver_version": "1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})
    for name in g.KERNEL_SWITCHES:
        monkeypatch.setenv(name, "0")
        monkeypatch.delenv(name)
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"dataset2": {"models": {KEY: entry()}}}))
    (tmp_path / "budget.yaml").write_text(yaml.safe_dump({"cap": {"money_usd": None, "gpu_hours": None},
                                                          "dataset2_allocation": {"discovery_units_per_arm": None,
                                                                                  "validation_units_per_arm": None}}))
    return tmp_path


def write_dec(path, **over):
    path.write_text(yaml.safe_dump(decoding(**over)))
    return path


def requests(path, seeds=True, n=3):
    rows = []
    for arm in ARMS3:
        for i in range(n):
            rid = f"c0|{arm}|{i}"
            r = {"request_id": rid, "model_key": KEY, "arm": arm, "config_id": "c0", "item_id": f"it{i}",
                 "messages_json": json.dumps([{"role": "user", "content": f"question {i}"}])}
            if seeds is True:
                r["sampling_seed"] = tc.seed_of(rid)
            elif isinstance(seeds, dict):
                r["sampling_seed"] = seeds[rid]
            rows.append(r)
    pd.DataFrame(rows).to_parquet(path)
    return path


def run(env, req, out, dec_path, **kw):
    return g.run_generation(req, KEY, out, backend="echo", models_path=env / "models.yaml", decoding_path=dec_path,
                            budget_path=env / "budget.yaml", verbose=False, shard_size=1000, **kw)


def outputs(folder):
    return pd.concat([pd.read_parquet(p) for p in sorted(Path(folder).glob("shard-*.parquet"))]).set_index("request_id")


# ---------------------------------------------------------------- the runner's sampling mode

def test_greedy_decodings_unchanged(tmp_path):
    d = g.load_decoding(write_dec(tmp_path / "d.yaml"))
    assert d.sampling == {} and "sampling" not in d.to_log()
    p = d.params_for_arm("direct_answer")
    assert p["temperature"] == 0.0 and "sampling_seed_source" not in p
    for f in ("decoding.yaml", "decoding_forced_math.yaml", "decoding_math_4k.yaml"):
        dd = g.load_decoding(ROOT / "config" / f)
        assert dd.temperature == 0.0 and not dd.sampling
        assert all("sampling_seed_source" not in dd.params_for_arm(a) for a in dd.arms)


@pytest.mark.parametrize("over,msg", [
    ({"temperature": 0.7}, "temperature must be 0"),
    ({"temperature": 0.0, "sampling": {"seed": "request"}}, "temperature in (0, 2]"),
    ({"temperature": 3.0, "sampling": {"seed": "request"}}, "temperature in (0, 2]"),
    ({"temperature": 0.7, "sampling": {"seed": 5}}, "sampling must be"),
    ({"temperature": True, "sampling": {"seed": "request"}}, "temperature in (0, 2]"),
])
def test_sampling_decoding_is_checked(tmp_path, over, msg):
    with pytest.raises(g.RunRefused) as e:
        g.load_decoding(write_dec(tmp_path / "d.yaml", **over))
    assert msg in str(e.value)


def test_sampled_decoding_params(tmp_path):
    d = g.load_decoding(write_dec(tmp_path / "d.yaml", **SAMPLED))
    assert d.temperature == 0.7 and d.to_log()["sampling"] == {"seed": "request"}
    p = d.params_for_arm("forced_direct")
    assert p["temperature"] == 0.7 and p["sampling_seed_source"] == "request" and p["prefill"] == "\\boxed{"


def test_seed_column_is_validated(env):
    ds = g.load_decoding(write_dec(env / "s.yaml", **SAMPLED))
    dg = g.load_decoding(write_dec(env / "g.yaml"))
    g.validate_requests(g.read_requests(requests(env / "r.parquet")), KEY, ds)
    with pytest.raises(g.RunRefused, match="no sampling_seed column"):
        g.validate_requests(g.read_requests(requests(env / "r2.parquet", seeds=False)), KEY, ds)
    with pytest.raises(g.RunRefused, match="decoding is greedy"):
        g.validate_requests(g.read_requests(requests(env / "r3.parquet")), KEY, dg)
    bad = {f"c0|{a}|{i}": (-1 if i == 0 else 5) for a in ARMS3 for i in range(3)}
    with pytest.raises(g.RunRefused, match="must be an integer"):
        g.validate_requests(g.read_requests(requests(env / "r4.parquet", seeds=bad)), KEY, ds)
    for s in (1.5, None, True, 2 ** 31):
        assert not g._is_seed(s)
    for s in (0, 7, np.int64(9), 3.0, 2 ** 31 - 1):
        assert g._is_seed(s)


def test_sampled_run_depends_on_seed_and_resumes(env):
    ds = write_dec(env / "s.yaml", **SAMPLED)
    dg = write_dec(env / "g.yaml")
    r1 = requests(env / "r1.parquet")
    log = run(env, r1, env / "o1", ds)
    assert log["status"] == "complete" and log["decoding"]["temperature"] == 0.7
    assert log["decoding"]["sampling"] == {"seed": "request"}
    a = outputs(env / "o1")
    # the same seeds give the same outputs; other seeds other outputs; greedy (no seed) the greedy echo output
    run(env, r1, env / "o2", ds)
    assert outputs(env / "o2")["output_text"].equals(a["output_text"])
    other = {f"c0|{x}|{i}": tc.seed_of(f"other|{x}|{i}") for x in ARMS3 for i in range(3)}
    run(env, requests(env / "r3.parquet", seeds=other), env / "o3", ds)
    assert (outputs(env / "o3")["output_text"] != a["output_text"]).all()
    run(env, requests(env / "rg.parquet", seeds=False), env / "og", dg)
    gr = outputs(env / "og")
    assert (gr["output_text"] != a["output_text"]).all()
    assert (gr["prompt_sha256"] == a["prompt_sha256"]).all()      # the prompts (prefill included) are the same
    # a resumed sampled run skips what is done and adds nothing
    log2 = run(env, r1, env / "o1", ds)
    assert log2["requests_this_run"] == 0
    # sampled and greedy folders never share a fingerprint
    assert log["run_fingerprint"] != json.loads(next((env / "og").glob("run_log-*.json")).read_text())["run_fingerprint"]


def test_sampled_greedy_echo_unchanged_without_seed(env):
    """The echo backend's greedy outputs do not move with the sampling change (no salt without a seed)."""
    dg = g.load_decoding(write_dec(env / "g.yaml"))
    be = g.EchoBackend(g.parse_model_spec(KEY, g.load_registry(env / "models.yaml")), dg)
    conv = [{"role": "user", "content": "q"}]
    p = dg.params_for_arm("direct_answer")
    r0 = be.generate([conv], [p])[0]
    r1 = be.generate([conv], [{**p, "sampling_seed": None}])[0]
    assert r0.text == r1.text


def test_vllm_backend_passes_temperature_and_seed(env, monkeypatch):
    seen = []
    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kw: seen.append(kw) or kw
    monkeypatch.setitem(sys.modules, "vllm", fake)

    class LLM:
        llm_engine = types.SimpleNamespace()

        def chat(self, convs, sps, **kw):
            return [types.SimpleNamespace(prompt=None, prompt_token_ids=[1], outputs=[types.SimpleNamespace(
                text="x", token_ids=[1], finish_reason="stop")]) for _ in convs]

    spec = g.parse_model_spec(KEY, g.load_registry(env / "models.yaml"))
    ds = g.load_decoding(write_dec(env / "s.yaml", **SAMPLED))
    dg = g.load_decoding(write_dec(env / "g.yaml"))
    be = g.make_backend("vllm", spec, ds)
    be.llm = LLM()
    conv = [{"role": "user", "content": "q"}]
    be.generate([conv], [{**ds.params_for_arm("direct_answer"), "sampling_seed": 123}])
    be.generate([conv], [dg.params_for_arm("direct_answer")])
    assert seen[0]["temperature"] == 0.7 and seen[0]["seed"] == 123
    assert seen[1]["temperature"] == 0.0 and "seed" not in seen[1]


# ---------------------------------------------------------------- the decoding files

@pytest.mark.parametrize("cap", [3072, 2900])
def test_decoding_files_differ_from_the_greedy_run_only_in_sampling(cap):
    diff = tc.params_differences(cap)
    for arm, v in diff.items():
        assert v["greedy_twin_equals_collection"], arm
        assert v["sampled_vs_collection"] == {"sampling_seed_source": ["request", None], "temperature": [0.7, 0.0]}
    d = g.load_decoding(ROOT / tc.DECODING[cap])
    assert d.max_new_tokens == {"direct_answer": 1024, "zero_shot_cot": cap, "forced_direct": 16}
    assert d.prefill == {"forced_direct": "\\boxed{"} and d.stop == () and d.top_p == 1.0 and d.top_k == 0


# ---------------------------------------------------------------- model rule, seeds, requests

def test_seed_rule():
    s = tc.seed_of("ps-i00-f05|direct_answer|test/algebra/2584.json")
    assert s == tc.seed_of("ps-i00-f05|direct_answer|test/algebra/2584.json") and 0 <= s < 2 ** 31
    assert s != tc.seed_of("ps-i00-f05|direct_answer|test/algebra/2584.json|fd")


def test_choose_models_rule():
    reg = {"a1": {"hf_id": "A1", "family": "A"}, "a2": {"hf_id": "A2", "family": "A"},
           "b1": {"hf_id": "B1", "family": "B"}, "b2": {"hf_id": "B2", "family": "B"}, "b3": {"hf_id": "B3", "family": "B"}}
    params = {"A1": 2.0, "A2": 9.0, "B1": 4.0, "B2": 6.0, "B3": 30.0}    # median 6.0
    r = tc.choose_models(list(reg), reg, params, families=("A", "B"))
    assert r["median_params_b"] == 6.0 and r["chosen"] == {"A": "a2", "B": "b2"}
    params.update({"A1": 5.0, "A2": 9.0, "B2": 7.0})     # median 7.0; A1 and A2 tie at distance 2: fewer parameters
    r = tc.choose_models(list(reg), reg, params, families=("A", "B"))
    assert r["chosen"]["A"] == "a1"


def test_saved_model_choice_follows_the_rule():
    man = ROOT / "manifests" / "temp_check_models.json"
    if not man.exists():
        pytest.skip("manifest absent")
    m = json.loads(man.read_text(encoding="utf-8"))
    assert [c["model_key"] for c in m["chosen"]] == ["gemma-3-12b-it", "llama-2-13b-chat-hf",
                                                     "mistral-nemo-instruct-2407", "phi-3-medium-4k-instruct",
                                                     "qwen3-14b"]
    assert m["median_params_b"] == 12.25 and m["n_panel"] == 29
    for fam, cands in m["candidates"].items():
        assert cands[0]["model_key"] in [c["model_key"] for c in m["chosen"]]
        assert cands == sorted(cands, key=lambda r: (r["distance"], r["params_b"], r["model_key"]))


def test_request_files_are_the_greedy_rows_plus_seeds():
    man = ROOT / "manifests" / "temp_check_requests.json"
    if not man.exists() or not (ROOT / "data" / "requests" / "temp_check" / "run").exists():
        pytest.skip("request files absent")
    m = json.loads(man.read_text(encoding="utf-8"))
    assert m["templates"] == ["ps-i00-f05", "ps-i02-f03", "ps-i03-f07", "ps-i06-f01"] and m["n_generations"] == 23880
    for e in m["models"]:
        df = pd.read_parquet(ROOT / e["files"]["run"]["path"])
        assert len(df) == 4776 and df["sampling_seed"].is_unique
        assert (df["sampling_seed"] == df["request_id"].map(tc.seed_of)).all()
        src = pd.concat([pd.read_parquet(ROOT / "data" / "requests" / "math" / f"{e['model_key']}.parquet"),
                         pd.read_parquet(ROOT / "data" / "requests" / "forced" / "math" / f"{e['model_key']}.parquet")])
        j = df.merge(src, on="request_id", suffixes=("", "_src"))
        assert len(j) == len(df) and (j["messages_json"] == j["messages_json_src"]).all()
        assert e["cot_cap"] == (2900 if e["model_key"] in ("llama-2-13b-chat-hf", "phi-3-medium-4k-instruct") else 3072)
        pil = pd.read_parquet(ROOT / e["files"]["pilot"]["path"])
        assert len(pil) == 24 and set(pil["config_id"]) == {"ps-i00-f05"}


# ---------------------------------------------------------------- the in-job gate

def _gate_work(tmp_path, *, greedy_same=24, sha_ok=True, temp=0.7):
    w = tmp_path / "work"
    ref_dir = w / "data" / "requests" / "temp_check" / "pilot_ref"
    ref_dir.mkdir(parents=True)
    rids = [f"c|{a}|{i}" for a in ARMS3 for i in range(8)]
    ref = pd.DataFrame({"request_id": rids, "prompt_sha256": [f"p{i}" for i in range(24)],
                        "output_text": [f"g{i}" for i in range(24)]})
    ref.to_parquet(ref_dir / "k.parquet")
    model = {"model_key": "k", "hf_id": "x"}
    vers = {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.16.1"}
    (ref_dir / "k.json").write_text(json.dumps({"model": model, "chat_template_sha256": "t", "gpu": "Ada",
                                                "versions": vers}))
    base = {"max_new_tokens": {"direct_answer": 1024, "forced_direct": 16, "zero_shot_cot": 3072},
            "prefill": {"forced_direct": "\\boxed{"}, "stop": [], "top_p": 1.0, "top_k": 0,
            "include_stop_str_in_output": False, "skip_special_tokens": True}
    for kind in ("pilot_greedy", "pilot"):
        d = w / "out" / kind / "k"
        d.mkdir(parents=True)
        out = pd.DataFrame({"request_id": rids, "arm": [r.split("|")[1] for r in rids],
                            "prompt_sha256": [f"p{i}" for i in range(24)] if sha_ok or kind == "pilot_greedy"
                            else [f"q{i}" for i in range(24)],
                            "output_text": ([f"g{i}" for i in range(greedy_same)] + ["other"] * (24 - greedy_same))
                            if kind == "pilot_greedy" else [f"s{i}" for i in range(24)],
                            "finish_reason": "stop", "n_output_tokens": 5})
        out.to_parquet(d / "shard-1-00000.parquet")
        dec = dict(base, temperature=0.0) if kind == "pilot_greedy" else dict(base, temperature=temp,
                                                                            sampling={"seed": "request"})
        (d / "run_log-1.json").write_text(json.dumps({
            "status": "complete", "model": model, "chat_template": {"observed_sha256": "t"},
            "prompt_render_mismatches": 0, "gpu": {"devices": [{"name": "Ada"}]}, "decoding": dec, "versions": vers}))
    return w


def test_gate_passes_and_fails(tmp_path):
    assert gate.gate("k", str(_gate_work(tmp_path / "a")))["passed"]
    assert gate.gate("k", str(_gate_work(tmp_path / "b", greedy_same=22)))["passed"]       # 22/24 >= 90%
    assert not gate.gate("k", str(_gate_work(tmp_path / "c", greedy_same=21)))["passed"]
    assert not gate.gate("k", str(_gate_work(tmp_path / "d", sha_ok=False)))["passed"]
    assert not gate.gate("k", str(_gate_work(tmp_path / "e", temp=1.0)))["passed"]
    w = _gate_work(tmp_path / "f")
    for f in (w / "out" / "pilot" / "k").glob("shard-*"):
        f.unlink()
    r = gate.gate("k", str(w))
    assert not r["passed"] and gate.main(["--model-key", "k", "--work", str(w)]) == 3


# ---------------------------------------------------------------- analysis

def _scored(rng, n_items=200, templates=("t1", "t2", "t3", "t4"), p=None, wlong=0.3, shift=0.0):
    p = p or {"direct_answer": 0.5, "forced_direct": 0.3, "zero_shot_cot": 0.7}
    rows = []
    for t in templates:
        for i in range(n_items):
            for arm, pa in p.items():
                pa = min(max(pa + (shift if arm == "zero_shot_cot" else 0.0), 0), 1)
                rows.append({"request_id": f"{t}|{arm}|{i}", "arm": arm, "config_id": t, "item_id": f"i{i:03d}",
                             "strict_correct": float(rng.random() < pa), "content_correct": float(rng.random() < pa),
                             "n_output_tokens": (100 if rng.random() < wlong else 5) if arm == "direct_answer" else 5,
                             "finish_reason": "stop", "output_text": "x"})
    return pd.DataFrame(rows)


def test_quantities_and_corrected_gain():
    q = tc.quantities(0.5, 0.3, 0.7, 0.25)
    assert q["req_gain"] == pytest.approx(0.2) and q["fd_gain"] == pytest.approx(0.4)
    assert q["corrected_gain"] == pytest.approx(0.75 * 0.4)


def test_analyse_recovers_known_differences():
    rng = np.random.default_rng(1)
    tmpl, items = ["t1", "t2", "t3", "t4"], [f"i{i:03d}" for i in range(200)]
    scored = {"m1": {"greedy": _scored(rng), "sampled": _scored(rng, shift=-0.2)},
              "m2": {"greedy": _scored(rng, wlong=0.1), "sampled": _scored(rng, wlong=0.1)}}
    an = tc.analyse(scored, tmpl, items, draws=400, seed=3)
    r = an["scores"]["strict_correct"]
    assert r["m1"]["diff"]["fd_gain"] == pytest.approx(-0.2, abs=0.05)
    lo, hi = r["m1"]["diff_ci"]["fd_gain"]
    assert lo < -0.2 < hi or abs(r["m1"]["diff"]["fd_gain"] + 0.2) < 0.05
    assert r["m2"]["diff_ci"]["w"][0] <= 0 <= r["m2"]["diff_ci"]["w"][1]
    for q in tc.QUANTS:
        assert r["pooled"]["diff"][q] == pytest.approx((r["m1"]["diff"][q] + r["m2"]["diff"][q]) / 2)
        assert r["pooled"]["sampled"][q] == pytest.approx((r["m1"]["sampled"][q] + r["m2"]["sampled"][q]) / 2)
    m = tc.mats(scored["m1"]["greedy"], "strict_correct", tmpl, items)
    assert set(m) == {"R", "F", "C", "W"} and m["R"].shape == (4, 200) and not np.isnan(m["W"]).any()
    assert r["m1"]["greedy"]["w"] == pytest.approx(float(np.mean(m["W"])))


def test_analyse_reports_missing_cells_without_imputing():
    rng = np.random.default_rng(2)
    tmpl, items = ["t1", "t2"], [f"i{i:03d}" for i in range(50)]
    s = _scored(rng, n_items=50, templates=tmpl)
    s2 = s.drop(index=s.index[:3])
    an = tc.analyse({"m": {"greedy": s, "sampled": s2}}, tmpl, items, draws=50, seed=1)
    # the first three rows are item 0's requested, forced and CoT rows: R, F, C and W (from the requested row) missing
    assert an["scores"]["strict_correct"]["m"]["sampled_missing_cells"] == 4
    assert an["scores"]["strict_correct"]["m"]["greedy_missing_cells"] == 0


def test_verify_greedy_scoring_counts_disagreements():
    a = pd.DataFrame({"model_key": "m", "arm": "direct_answer", "config_id": "c", "item_id": ["1", "2", "3"],
                      "strict_correct": [1.0, 0.0, 1.0], "content_correct": [1.0, 1.0, 0.0]})
    p = a.copy()
    p.loc[1, "strict_correct"] = 1.0
    v = tc.verify_greedy_scoring(a, p)
    assert v["strict_correct"]["disagree"] == 1 and v["content_correct"]["disagree"] == 0


def test_sacct_parsing_and_job_summary(tmp_path):
    txt = "101|d2-tc-p-a|COMPLETED|600|0:0|n1|s|e\n102|d2-tc-r-a|CANCELLED by 5|0|0:0|None|s|e\n"
    s = tc.parse_sacct(txt)
    assert s["101"]["elapsed_s"] == 600 and s["102"]["state"] == "CANCELLED"
    js = tc.job_summary([{"model_key": "a", "kind": "pilot", "job_id": "101", "name": "d2-tc-p-a"},
                         {"model_key": "a", "kind": "run", "job_id": "102", "name": "d2-tc-r-a"}], s)
    assert js["all_final"] and js["scheduler_gpu_hours"] == pytest.approx(600 / 3600, abs=1e-3)
    pending = tc.job_summary([{"model_key": "a", "kind": "run", "job_id": "103", "name": "x"}], s)
    assert not pending["all_final"]


def test_pending_report_names_the_submit_command():
    md = tc.render_report(None, {"jobs": [], "scheduler_gpu_hours": None, "all_final": False}, None, None)
    assert "scripts/temp_check.py build" in md and md.startswith("# Temperature 0.7 spot check")


# ---------------------------------------------------------------- macros

def _results(tmp_path):
    rng = np.random.default_rng(4)
    tmpl, items = ["t1", "t2", "t3", "t4"], [f"i{i:03d}" for i in range(60)]
    keys = ["gemma-3-12b-it", "llama-2-13b-chat-hf", "mistral-nemo-instruct-2407", "phi-3-medium-4k-instruct",
            "qwen3-14b"]
    scored = {k: {"greedy": _scored(rng, n_items=60), "sampled": _scored(rng, n_items=60, shift=-0.05)} for k in keys}
    an = tc.analyse(scored, tmpl, items, draws=100, seed=5)
    fam = dict(zip(keys, ("Gemma", "Llama", "Mistral", "Phi", "Qwen")))
    for d in scored.values():
        d["greedy"]["model_key"] = "x"
    R = {"design": {"models": [{"family": fam[k], "model_key": k, "name": k} for k in keys], "templates": tmpl,
                    "n_items": 60, "sampling": {"temperature": 0.7}, "n_generations_sampled": 23880},
         "analysis": an, "outputs": tc.output_stats(scored), "jobs": {"scheduler_gpu_hours": 2.345}}
    p = tmp_path / "results" / "temp_check"
    p.mkdir(parents=True)
    (p / "temp_check.json").write_text(json.dumps(R, default=tc._default))
    return tmp_path


def test_macros(tmp_path):
    assert mtc.temp_check_macros(mtc.temp_check_results(str(tmp_path))) == {}
    root = _results(tmp_path)
    M = mtc.temp_check_macros(mtc.temp_check_results(str(root)))
    assert all(k.startswith("tempCheck") and k.isalpha() for k in M)
    for L in ("Gemma", "Llama", "Mistral", "Phi", "Qwen", "Pooled"):
        for k in (f"tempCheckWMath{L}Pct", f"tempCheckWDiffMath{L}PctLo", f"tempCheckReqGainDiffMath{L}",
                  f"tempCheckFdGainDiffMathContent{L}Hi", f"tempCheckCorrGainGreedyMath{L}"):
            assert k in M, k
    assert M["tempCheckGenerations"] == "23{,}880" and M["tempCheckTemperature"] == "0.7"
    assert M["tempCheckGpuHours"] == "2.35" and M["tempCheckModels"] == "5"
    assert all("-" not in v or "$-$" in v for v in M.values() if v[:1] in "-$0123456789")
    assert mtc.num(-0.04) == "0.0" and mtc.num(-1.25, 1) in ("$-$1.2", "$-$1.3")


def test_macro_names_are_new():
    nt = ROOT / "paper" / "numbers.tex"
    if not nt.exists():
        pytest.skip("numbers.tex absent")
    text = nt.read_text(encoding="utf-8")
    if "tempcheck" in (ROOT / "paper" / "make_numbers.py").read_text(encoding="utf-8").lower():
        # wired: names must then be defined exactly once
        import re
        names = re.findall(r"\\newcommand\{\\(tempCheck[A-Za-z]*)\}", text)
        assert names and len(names) == len(set(names))
        return
    assert "\\newcommand{\\tempCheck" not in text
