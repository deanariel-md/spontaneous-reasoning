"""Native reasoning under the vendors' sampling settings (sample rule, decoding file, scoring and analysis)."""
import hashlib
import json
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_thinking_vendor_requests as bv  # noqa: E402
import run_thinking_vendor as rv  # noqa: E402
import score_thinking_vendor as sv  # noqa: E402
import thinking_vendor_generate as tg  # noqa: E402


def test_seed_is_the_hash_prefix():
    want = int.from_bytes(hashlib.sha256(b"d20/vendor/gsm8k").digest()[:8], "big")
    assert bv.seed_for("gsm8k") == want
    assert bv.seed_for("math") != want


def test_draw_is_fixed_sorted_and_a_subset():
    ids = [str(i) for i in range(1319)]
    a, b = bv.draw_items(ids, "gsm8k"), bv.draw_items(list(reversed(ids)), "gsm8k")
    assert a == b and len(a) == 300 and len(set(a)) == 300
    assert a == sorted(a, key=int) and set(a) <= set(ids)


def test_gsm8k_ids_sort_as_integers_and_math_ids_as_strings():
    assert bv.sort_ids(["10", "2", "1"], "gsm8k") == ["1", "2", "10"]
    assert bv.sort_ids(["b/10", "b/2", "a/1"], "math") == ["a/1", "b/10", "b/2"]


def test_too_few_items_refuse():
    with pytest.raises(ValueError):
        bv.draw_items(["1", "2"], "gsm8k", n=3)


def test_first_configs_skip_repeated_draws():
    assert bv.first_configs({"discovery_config_ids": ["a", "a", "b", "c"]}) == ["a", "b"]
    with pytest.raises(ValueError):
        bv.first_configs({"discovery_config_ids": ["a", "a"]})


def test_the_saved_sample_follows_the_rule():
    p = os.path.join(ROOT, "manifests", "thinking_vendor_sample.json")
    if not os.path.exists(p):
        pytest.skip("sample not built")
    man = json.load(open(p, encoding="utf-8"))
    assert man["models"] == ["qwen3.5-35b-a3b", "qwen3.6-27b", "qwen3.6-35b-a3b"]
    for t in ("gsm8k", "math"):
        s = json.load(open(os.path.join(ROOT, "manifests", f"sampler_{t}.json"), encoding="utf-8"))
        m = man["tasks"][t]
        assert m["configurations"] == bv.first_configs(s)
        assert len(m["items"]) == 300 and m["items_sha256"] == bv.list_sha(m["items"])
        assert m["seed"] == bv.seed_for(t)


def test_vendor_file_holds_the_cards_settings():
    v = tg.load_vendor()
    assert v["max_new_tokens"] == 30720 and v["seed"] == 0 and v["skip_special_tokens"] is False
    for k in ("qwen3.5-35b-a3b", "qwen3.6-35b-a3b", "qwen3.6-27b"):
        kw = tg.sampling_kwargs(v, k)
        assert (kw["temperature"], kw["top_p"], kw["top_k"], kw["min_p"]) == (1.0, 0.95, 20, 0.0)
        assert kw["seed"] == 0 and kw["max_tokens"] == 30720 and kw["stop"] is None
    assert tg.sampling_kwargs(v, "qwen3.5-35b-a3b")["presence_penalty"] == 1.5
    assert tg.sampling_kwargs(v, "qwen3.6-35b-a3b")["presence_penalty"] == 1.5
    assert tg.sampling_kwargs(v, "qwen3.6-27b")["presence_penalty"] == 0.0


def test_vendor_card_hashes_match_the_kept_cards():
    v = tg.load_vendor()
    for k, m in v["models"].items():
        d = os.path.join(ROOT, "manifests", "external", "vendor_cards", k)
        g = os.path.join(d, "generation_config.json")
        assert hashlib.sha256(open(g, "rb").read()).hexdigest() == m["source"]["generation_config_sha256"]
        gc = json.load(open(g, encoding="utf-8"))
        assert (gc["temperature"], gc["top_p"], gc["top_k"]) == (m["temperature"], m["top_p"], m["top_k"])


def test_greedy_settings_refused(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump({"status": "confirmed", "models": {"x": {s: 0.0 for s in tg.SAMPLING_KEYS}}}))
    with pytest.raises(SystemExit):
        tg.load_vendor(p)


def test_item_i_entry_is_shared_by_the_two_tasks():
    e = tg.load_entry("qwen3.6-27b")
    assert e["thinking_off"]["chat_template_kwargs"]["enable_thinking"] is True
    assert tg.engine_settings("qwen3.6-27b")["enable_prefix_caching"] is True


def test_parts_cover_every_row_once():
    df = pd.DataFrame({"request_id": [f"r{i}" for i in range(11)]})
    got = pd.concat([tg.select_part(df, f"{i}/3") for i in range(3)])
    assert sorted(got["request_id"]) == sorted(df["request_id"])


def test_flags_read_after_the_marker():
    loop = "\n".join(["same line"] * 120)
    fl = sv.flags(["think</think>\n\\boxed{4}", "still thinking", loop, "no marker \\boxed{3}"],
                  ["stop", "length", "length", "stop"])
    assert fl["closed"].tolist() == [True, False, False, False]
    assert fl["cut_open"].tolist() == [False, True, True, False]
    assert fl["looping"].tolist() == [False, False, True, False]
    assert fl["body"].tolist()[:2] == ["\n\\boxed{4}", ""]
    assert fl["body"].tolist()[3] == "no marker \\boxed{3}"


def _req():
    return pd.DataFrame({"task": ["gsm8k", "gsm8k"], "request_id": ["c|d|1|th|tv", "c|d|2|th|tv"],
                         "item_i_request_id": ["c|d|1|th", "c|d|2|th"], "model_key": "m", "arm": "thinking_vendor_gsm8k",
                         "config_id": "c", "item_id": ["1", "2"], "messages_json": "[]", "prompt_sha256": "x",
                         "n_prompt_tokens": 5})


def _out(ids, texts):
    return pd.DataFrame({"request_id": ids, "output_text": texts, "n_output_tokens": [10] * len(ids),
                         "finish_reason": ["stop"] * len(ids)})


def test_assemble_pairs_vendor_and_greedy_rows():
    rows, missing = sv.assemble(_req(), _out(["c|d|1|th|tv", "c|d|2|th|tv"], ["v1", "v2"]),
                                _out(["c|d|1|th", "c|d|2|th", "other"], ["g1", "g2", "g3"]))
    assert missing == 0
    assert rows.groupby("decoding")["output_text"].apply(list).to_dict() == {"greedy": ["g1", "g2"], "vendor": ["v1", "v2"]}


def test_assemble_refuses_missing_vendor_outputs_unless_allowed():
    with pytest.raises(SystemExit):
        sv.assemble(_req(), _out(["c|d|1|th|tv"], ["v1"]), _out(["c|d|1|th", "c|d|2|th"], ["g1", "g2"]))
    rows, missing = sv.assemble(_req(), _out(["c|d|1|th|tv"], ["v1"]), _out(["c|d|1|th", "c|d|2|th"], ["g1", "g2"]),
                                allow_missing=True)
    assert missing == 1 and (rows["decoding"] == "vendor").sum() == 1


def test_item_bootstrap_keeps_an_items_cells_together():
    r = rv.item_bootstrap([5.0, 5.0, 5.0, 5.0], ["a", "a", "b", "b"], n_boot=200)
    assert r["mean"] == r["lo"] == r["hi"] == 5.0 and r["n_items"] == 2 and r["n_cells"] == 4
    r = rv.item_bootstrap([0.0, 100.0] * 50, [str(i) for i in range(50) for _ in range(2)], n_boot=500)
    assert r["lo"] == r["hi"] == 50.0                    # each item's two cells average 50: no item-level spread
    assert rv.item_bootstrap([], [])["mean"] is None


def test_share_reports_an_empty_denominator_as_none():
    assert rv.share(0, 0)["pct"] is None
    s = rv.share(3, 10)
    assert s["pct"] == 30.0 and s["lo"] < 30 < s["hi"]


def _scored(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in ("m1", "m2"):
        for t in ("gsm8k", "math"):
            for c in ("c1", "c2"):
                for i in range(30):
                    for d in ("vendor", "greedy"):
                        cut = bool(rng.random() < (0.05 if d == "vendor" else 0.3))
                        rows.append({"task": t, "model_key": k, "config_id": c, "item_id": str(i), "decoding": d,
                                     "request_id": f"{c}|{i}", "n_output_tokens": 30720 if cut else 2000,
                                     "finish_reason": "length" if cut else "stop", "closed": not cut,
                                     "cut_open": cut, "looping": cut and bool(rng.random() < 0.7),
                                     "strict_correct": int((not cut) and rng.random() < 0.9),
                                     "lenient_correct": int((not cut) and rng.random() < 0.92)})
    return pd.DataFrame(rows)


def test_run_on_synthetic_scores(tmp_path, monkeypatch):
    p = tmp_path / "scored.parquet"
    _scored().to_parquet(p, index=False)
    sample = {"tasks": {t: {"configurations": ["c1", "c2"], "items": [str(i) for i in range(30)]} for t in ("gsm8k", "math")}}
    sp = tmp_path / "sample.json"
    sp.write_text(json.dumps(sample))
    monkeypatch.setattr(rv, "SAMPLE", str(sp))
    res = rv.run(Namespace(scored=str(p), out=str(tmp_path / "out"), no_comparators=True))
    blk = res["cells"]["gsm8k/m1"]
    assert blk["vendor"]["n"] == 60 and blk["greedy"]["n"] == 60
    assert blk["vendor"]["cut"]["pct"] < blk["greedy"]["cut"]["pct"]
    d = blk["diff_vendor_minus_greedy"]
    assert d["n_items"] == 30 and d["lo"] <= d["mean"] <= d["hi"]
    s = _scored()
    sub = s[(s["model_key"] == "m1") & (s["task"] == "gsm8k")]
    want = 100 * (sub[sub["decoding"] == "vendor"]["strict_correct"].mean() - sub[sub["decoding"] == "greedy"]["strict_correct"].mean())
    assert d["mean"] == pytest.approx(want)
    for f in ("thinking_vendor.json", "summary.md", "summary_clean.md", "per_cell.csv"):
        assert (tmp_path / "out" / f).exists()


def test_built_request_files_match_item_i():
    rec_p = os.path.join(ROOT, "data", "requests", "thinking_vendor", "build_record.json")
    if not os.path.exists(rec_p):
        pytest.skip("requests not built")
    rec = json.load(open(rec_p, encoding="utf-8"))
    man = json.load(open(os.path.join(ROOT, "manifests", "thinking_vendor_sample.json"), encoding="utf-8"))
    for k, ent in rec["models"].items():
        df = pd.read_parquet(os.path.join(ROOT, "data", "requests", "thinking_vendor", ent["requests_file"]))
        assert len(df) == ent["n"] == 1200 and df["request_id"].is_unique
        for t in ("gsm8k", "math"):
            sub = df[df["task"] == t]
            assert set(sub["config_id"]) == set(man["tasks"][t]["configurations"])
            assert set(sub["item_id"]) == set(man["tasks"][t]["items"])
            src = pd.read_parquet(os.path.join(ROOT, "data", "requests", "thinking", t, f"{k}.parquet"))
            m = sub.merge(src, left_on="item_i_request_id", right_on="request_id", suffixes=("", "_i"))
            assert len(m) == 600 and (m["messages_json"] == m["messages_json_i"]).all()
