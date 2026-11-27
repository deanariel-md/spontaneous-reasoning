"""The forced-direct arm: an assistant prefill per arm (src/generation.py)."""
import pandas as pd
import pytest

from src import generation as g
from tests.test_generation import Cfg, KEY, decoding_dict, fi, kernel_switches, make_requests  # noqa: F401

FORCED = decoding_dict(max_new_tokens={"forced_direct": 16}, prefill={"forced_direct": "\\boxed{"}, stop=["}"],
                       include_stop_str_in_output=True)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 1, "devices": [{"name": "Fake A100", "memory_mib": 81920}], "driver_version": "580.1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})
    return Cfg(tmp_path, decoding=FORCED)


def test_prefill_enters_params_only_where_set():
    plain = g.load_decoding(g.DECODING_YAML)
    assert all("prefill" not in plain.params_for_arm(a) for a in plain.arms)   # collection cache keys unchanged
    assert plain.prefill == {}


def test_forced_decoding_file_parses():
    d = g.load_decoding(g.ROOT / "config" / "decoding_forced.yaml")
    assert d.params_for_arm("forced_direct")["prefill"] == "\\boxed{"
    assert d.stop == ("}",) and d.include_stop_str_in_output and d.max_new_tokens == {"forced_direct": 16}
    n = g.load_decoding(g.ROOT / "config" / "decoding_forced_noprefill.yaml")
    assert "prefill" not in n.params_for_arm("forced_direct")


def test_prefill_for_an_unknown_arm_is_refused(tmp_path):
    p = tmp_path / "d.yaml"
    import yaml
    p.write_text(yaml.safe_dump(decoding_dict(prefill={"other": "x"})))
    with pytest.raises(g.RunRefused, match="prefill names arms"):
        g.load_decoding(p)


def test_echo_run_renders_the_open_assistant_turn(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", arms=("forced_direct",), n_items=2, configs=("c0",), same_as={})
    be = cfg.echo()
    seen = []
    orig = be.generate

    def spy(convs, params):
        seen.extend(zip(convs, params))
        return orig(convs, params)

    be.generate = spy
    log = cfg.run(req, tmp_path / "o", backend_obj=be)
    assert log["status"] == "complete"
    for conv, p in seen:
        assert conv[-1] == {"role": "assistant", "content": "\\boxed{"} and p["prefill"] == "\\boxed{"
        assert be.render(conv).endswith("<|assistant|>\n\\boxed{")
    out = pd.concat(pd.read_parquet(f) for f in (tmp_path / "o").glob("shard-*.parquet"))
    assert len(out) == 2 and set(out["arm"]) == {"forced_direct"}
