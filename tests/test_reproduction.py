"""Reproduction of Cheng et al. (2025): scorer, requests, intervals, analysis and the fixed configuration files."""
import json
import math
import random
import re

import pandas as pd
import pytest
import yaml

from src import generation as gen, reproduction as rp

ROOT = rp.ROOT


# The authors' released functions (opencompass/datasets/gsm8k.py at commit 40de49b9006c of
# github.com/small-xiangcheng/opencompass, Apache-2.0), copied verbatim as the oracle for src/reproduction.py.
def _o_last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval


def _o_remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def _o_extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = _o_last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = _o_remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\\{(.*)\\}$', answer)
        if match:
            answer = match.group(1)
    return answer


def _o_gsm8k_postprocess_fixed(text):
    text = text.replace(',', '')
    cand_ans = _o_extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', cand_ans)
        if not numbers:
            return 'NULL'
        return numbers[-1]
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]


def _o_is_equal(pred, refer):
    try:
        if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
            return True
    except Exception:
        pass
    return False


CASES = [
    ("There are 15 trees. So the answer is $\\boxed{6}$.", "6"),
    ("Total \\boxed{1,000} dollars", "1000"),
    ("She has \\boxed{\\$18} left", "18"),
    ("\\boxed{{42}}", "42"),
    ("The answer is \\boxed{x}", "NULL"),
    ("5 apples \\boxed{} then", "5"),
    ("Five apples \\boxed{} then", "NULL"),
    ("The answer is 12.\n\nQuestion: next 99", "12"),
    ("\\fbox{7}", "7"),
    ("Unclosed \\boxed{12", "12"),
    ("First \\boxed{3} then \\boxed{-4.5}", "-4.5"),
    ("no numbers at all", "NULL"),
    ("\\boxed{\\frac{1}{2}}", "2"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_extract_cases(text, expected):
    assert rp.cheng_extract(text) == expected == _o_gsm8k_postprocess_fixed(text)


def test_extract_matches_authors_code_on_random_strings():
    rng = random.Random(20261031)
    alphabet = list("0123456789,.-{} xQuestion:\\boxedfbx$") + ["\\boxed{", "\\fbox{", "Question:", "}", "{"]
    for _ in range(5000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert rp.cheng_extract(s) == _o_gsm8k_postprocess_fixed(s), s


@pytest.mark.parametrize("pred,ref", [("5", "5"), ("18.0", "18"), ("2.50", "2"), ("NULL", "5"), ("-3", "-3"),
                                      ("1e3", "1000"), ("7", "07"), ("abc", "abc")])
def test_is_equal_matches_authors_code(pred, ref):
    assert rp.cheng_is_equal(pred, ref) == _o_is_equal(pred, ref)


def test_reference():
    assert rp.cheng_reference("Step one.\n#### 1,234") == "1234"


# --------------------------------------------------------------------------------------------
# Configuration files fixed with the protocol
# --------------------------------------------------------------------------------------------

def test_config_and_registry_agree():
    cfg = rp.load_config()
    reg = gen.load_registry(ROOT / cfg["registry"])
    assert list(reg) == list(cfg["models"])
    for key in reg:
        spec = gen.parse_model_spec(key, reg)
        assert spec.chat_template.sha256 is not None
        assert spec.kernels == "batch_invariant"
        fp8 = cfg["models"][key].get("precision") == "fp8"
        assert (spec.quantization == "compressed-tensors") == fp8
        if spec.chat_template.source == "file":
            text = (ROOT / spec.chat_template.file).read_text(encoding="utf-8")
            assert not gen.template_reads_clock(text)
            assert gen.template_text_sha256(text) == spec.chat_template.sha256
            assert spec.revision in text


def test_decoding_files_and_llama2_caps():
    cfg = rp.load_config()
    prompts = rp.load_json(cfg["prompts"])
    reg = gen.load_registry(ROOT / cfg["registry"])
    for key in cfg["models"]:
        path = cfg["decoding"]["by_model"].get(key, cfg["decoding"]["default"])
        dec = gen.load_decoding(ROOT / path)
        assert dec.confirmed and dec.arms == tuple(sorted(rp.ARMS))
        assert list(dec.stop) == prompts["stop"]
        mml = reg[key]["max_model_len"]
        if key == "llama-2-7b-chat-hf":
            assert mml == 4096
            assert dec.max_new_tokens["zero_shot_cot"] + 256 <= mml and dec.max_new_tokens["eight_shot_cot"] + 1408 <= mml
        else:
            assert dec.max_new_tokens == {"zero_shot_cot": 4096, "eight_shot_cot": 4096} and mml >= 8192


def test_prompt_manifest():
    p = rp.load_json("manifests/cheng_exemplars.json")
    assert len(p["exemplars"]) == 8 and p["max_out_len"] == 4096
    assert all(e["answer"].rstrip("\n").endswith("$.") and "\\boxed{" in e["answer"] for e in p["exemplars"])
    assert p["zero_shot_user"] == "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
    assert p["eight_shot_user"].startswith("Question: {question}\n") and p["eight_shot_user"].endswith("\nAnswer:")


# --------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------

ITEMS = pd.DataFrame({"item_id": [0, 1, 2], "question": ["Q zero?", "Q {one}?", "Q two?"]})


def test_fill_refuses_other_counts():
    with pytest.raises(ValueError):
        rp.fill([{"role": "user", "content": "no slot"}], "x")
    with pytest.raises(ValueError):
        rp.fill([{"role": "user", "content": "{question} {question}"}], "x")


def test_numerical_requests():
    cfg = rp.load_config()
    prompts = rp.load_json(cfg["prompts"])
    arms = rp.part_arms("numerical", cfg, prompts)
    df = rp.build_requests("m", arms, ITEMS)
    assert len(df) == 6 and list(df.columns) == list(gen.REQUEST_COLUMNS)
    assert set(df["config_id"]) == {"cheng"} and df["request_id"].is_unique
    zs = json.loads(df.loc[df["request_id"] == "cheng|zero_shot_cot|1", "messages_json"].item())
    assert zs == [{"role": "user", "content": prompts["zero_shot_user"].replace("{question}", "Q {one}?")}]
    es = json.loads(df.loc[df["request_id"] == "cheng|eight_shot_cot|2", "messages_json"].item())
    assert len(es) == 17 and [m["role"] for m in es] == ["user", "assistant"] * 8 + ["user"]
    assert [m["content"] for m in es[1:16:2]] == [e["answer"] for e in prompts["exemplars"]]
    assert es[0]["content"] == prompts["eight_shot_user"].replace("{question}", prompts["exemplars"][0]["question"])
    assert es[-1]["content"] == prompts["eight_shot_user"].replace("{question}", "Q two?")


def test_transport_requests():
    cfg = rp.load_config()
    prompts = rp.load_json(cfg["prompts"])
    bank = rp.load_json(cfg["parts"]["transport"]["bank"])
    tranche = rp.load_json(cfg["parts"]["transport"]["configurations_from"])
    arms = rp.part_arms("transport", cfg, prompts, bank)
    assert list(arms) == tranche["configurations"] and len(arms) == 8
    df = rp.build_requests("m", arms, ITEMS)
    assert len(df) == 8 * 2 * 3
    cid = tranche["configurations"][3]
    frame = bank["configurations"][cid]["arms"]["zero_shot_cot"]["messages"][0]["content"]
    zs = json.loads(df.loc[df["request_id"] == f"{cid}|zero_shot_cot|0", "messages_json"].item())
    assert zs == [{"role": "user", "content": frame.replace("{question}", "Q zero?")}]
    es = json.loads(df.loc[df["request_id"] == f"{cid}|eight_shot_cot|0", "messages_json"].item())
    assert len(es) == 17
    assert es[2]["content"] == frame.replace("{question}", prompts["exemplars"][1]["question"])
    assert es[3]["content"] == prompts["exemplars"][1]["answer"]
    assert es[-1]["content"] == frame.replace("{question}", "Q zero?")


def test_eight_shot_refuses_multi_turn_frames():
    with pytest.raises(ValueError):
        rp.eight_shot([{"role": "system", "content": "s"}, {"role": "user", "content": "{question}"}],
                      [{"question": "q", "answer": "a"}] * 8)


# --------------------------------------------------------------------------------------------
# Intervals and analysis
# --------------------------------------------------------------------------------------------

def test_wilson_known_values():
    lo, hi = rp.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = rp.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4) and hi == pytest.approx(1.0, abs=1e-12)
    assert all(math.isnan(v) for v in rp.wilson(0, 0))


@pytest.mark.parametrize("a,b,c,d", [(12, 9, 2, 21), (500, 30, 45, 744), (1300, 0, 0, 19), (0, 5, 0, 0), (40, 1, 7, 3)])
def test_newcombe_properties(a, b, c, d):
    r = rp.newcombe_paired(a, b, c, d)
    n = a + b + c + d
    assert r["diff"] == pytest.approx((b - c) / n)
    assert -1 <= r["lo"] <= r["diff"] <= r["hi"] <= 1
    s = rp.newcombe_paired(a, c, b, d)
    assert s["diff"] == pytest.approx(-r["diff"]) and s["lo"] == pytest.approx(-r["hi"]) and s["hi"] == pytest.approx(-r["lo"])


def test_newcombe_empty_margin_is_recorded():
    r = rp.newcombe_paired(0, 5, 0, 0)
    assert r["phi"] == 0.0 and r["phi_note"]


def _scored(rows):
    df = pd.DataFrame(rows, columns=["model_key", "part", "config_id", "arm", "item_id", "correct"])
    df["null_pred"] = False
    df["truncated"] = False
    return df


def _model_rows(m, part, cid, zero, eight):
    rows = []
    for i, (z, e) in enumerate(zip(zero, eight)):
        rows += [(m, part, cid, "zero_shot_cot", i, z), (m, part, cid, "eight_shot_cot", i, e)]
    return rows


CFG = {"interval_level": 0.95, "models": {
    "s1": {"group": "strong", "zero_shot": 90.0, "eight_shot": 89.0, "source": "x"},
    "s2": {"group": "strong", "zero_shot": 80.0, "eight_shot": 79.0, "source": "x", "precision": "fp8"},
    "w1": {"group": "weak", "zero_shot": 30.0, "eight_shot": 40.0, "source": "x"}}}


def test_analyze_claim_signs_and_slopes():
    n = 200
    rows = []
    rows += _model_rows("s1", "numerical", "cheng", [1] * 180 + [0] * 20, [1] * 178 + [0] * 22)
    rows += _model_rows("s2", "numerical", "cheng", [1] * 100 + [0] * 100, [1] * 100 + [1] * 60 + [0] * 40)
    rows += _model_rows("w1", "numerical", "cheng", [1] * 60 + [0] * 140, [1] * 60 + [1] * 40 + [0] * 100)
    for cid in ("c1", "c2"):
        rows += _model_rows("s1", "transport", cid, [1] * 180 + [0] * 20, [1] * 180 + [0] * 20)
        rows += _model_rows("w1", "transport", cid, [1] * 60 + [0] * 140, [1] * 70 + [0] * 130)
    s = rp.analyze(_scored(rows), CFG)
    A = s["numerical"]
    assert A["claim_bf16"]["holds"] is True
    assert A["claim_with_fp8"]["holds"] is False and A["claim_with_fp8"]["strong_models_with_benefit_above_zero"] == ["s2"]
    ben = {r["model_key"]: r for r in A["benefit"]}
    assert ben["s1"]["benefit"] == pytest.approx(-1.0) and ben["s1"]["sign_agrees"] is True
    assert ben["s2"]["sign_agrees"] is False and ben["w1"]["benefit"] == pytest.approx(20.0)
    assert A["models_not_run"] == []
    B = s["transport"]
    assert B["configurations"] == ["c1", "c2"] and B["claim_holds_in"] == {"bf16": 2, "with_fp8": 2, "of": 2}
    assert B["per_model_sign_agreement"]["s1"] == {"agree": 0, "disagree": 0, "configurations": 2}  # ties
    assert B["equal_budget_slope"]["n_models"] == 2 and B["equal_budget_slope"]["sign"] == -1
    assert n == 200


def test_capability_slope_edge_cases():
    assert rp.capability_slope([1.0], [50.0])["note"] == "fewer than two models"
    assert rp.capability_slope([1.0, 2.0], [50.0, 50.0])["note"] == "no spread in zero-shot accuracy"
    r = rp.capability_slope([1.0, 1.0], [40.0, 60.0])
    assert r["slope"] == 0 and r["sign"] == 0 and r["note"] == "tie"


def test_score_outputs_flags():
    items = pd.DataFrame({"item_id": [0, 1], "question": ["a", "b"], "gold_text": ["x\n#### 6", "y\n#### 1,000"]})
    out = pd.DataFrame({"request_id": ["cheng|zero_shot_cot|0", "cheng|zero_shot_cot|1"], "model_key": ["m", "m"],
                        "arm": ["zero_shot_cot"] * 2, "config_id": ["cheng"] * 2, "item_id": [0, 1],
                        "output_text": ["So the answer is $\\boxed{6}$.", "words only"], "n_output_tokens": [10, 4096],
                        "finish_reason": ["stop", "length"], "cached": [False, False]})
    s = rp.score_outputs(out, items, "numerical")
    assert s["correct"].tolist() == [1, 0] and s["null_pred"].tolist() == [False, True]
    assert s["truncated"].tolist() == [False, True] and set(s["scorer"]) == {rp.SCORER_VERSION}


def test_registry_is_separate_from_the_panel():
    panel = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["dataset2"]["models"]
    repro = gen.load_registry(ROOT / "config" / "models_reproduction.yaml")
    for k in set(panel) & set(repro):
        assert panel[k]["revision"] == repro[k]["revision"]
        assert panel[k]["chat_template"]["sha256"] == repro[k]["chat_template"]["sha256"]
