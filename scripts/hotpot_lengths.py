r"""Token lengths of every HotpotQA item for the length rule and the forced cap.

Run in a Python environment with `transformers` (the project venv has none), from the repository root, reading the pinned raw file as data:
    <tokenizer venv>\Scripts\python.exe -I scripts\hotpot_lengths.py

Writes data/processed/hotpot/lengths.parquet and lengths.json (its sha256 and the tokenizer revisions):
  item_tokens  the item text (src/hotpot_task.item_text), maximum over the pinned tokenizers of the four panel
               checkpoints with a native context of 4,096 (hotpot_task.LENGTH_MODELS), no special tokens
  demo_tokens  the demonstration text with its gold answer (hotpot_task.texts, demo_text), the same maximum
  gold_tokens  '\text{<gold>}}' (the forced arm's longest plausible answer), maximum over the 29 panel tokenizers
Tokenizers are read from the local Hugging Face cache at each model's pinned revision (HF_HUB_OFFLINE=1): this
script downloads nothing. Nothing here reads a model output.
"""
import hashlib
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import hotpot_task as ht  # noqa: E402

REGISTRIES = (os.path.join(ROOT, "config", "models.yaml"), os.path.join(ROOT, "config", "models_extension.yaml"))
PANEL = os.path.join(ROOT, "results", "dataset2", "gsm8k30.json")


def registry() -> dict:
    out = {}
    for p in REGISTRIES:
        out.update({k: v for k, v in yaml.safe_load(open(p, encoding="utf-8"))["dataset2"]["models"].items()
                    if not k.startswith("_")})
    return out


def main() -> int:
    from transformers import AutoTokenizer
    t0 = time.time()
    raw = ht.load_raw()
    tx = ht.texts(raw)
    reg = registry()
    panel = json.load(open(PANEL, encoding="utf-8"))["models"]
    toks, pins = {}, {}
    for k in panel:
        e = reg[k]
        toks[k] = AutoTokenizer.from_pretrained(e["hf_id"], revision=e["revision"], trust_remote_code=False)
        pins[k] = {"hf_id": e["hf_id"], "revision": e["revision"], "class": type(toks[k]).__name__}

    def n(tok, s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    item = np.zeros(len(tx), dtype=np.int64)
    demo = np.zeros(len(tx), dtype=np.int64)
    for k in ht.LENGTH_MODELS:
        item = np.maximum(item, [n(toks[k], s) for s in tx["item_text"]])
        demo = np.maximum(demo, [n(toks[k], s) for s in tx["demo_text"]])
        print(f"{k}: done ({time.time() - t0:.0f} s)", flush=True)
    gold = np.zeros(len(tx), dtype=np.int64)
    for k in panel:
        gold = np.maximum(gold, [n(toks[k], "\\text{" + g + "}}") for g in tx["gold_answer"]])
    out = pd.DataFrame({"item_id": tx["item_id"], "item_tokens": item.astype("int64"),
                        "demo_tokens": demo.astype("int64"), "gold_tokens": gold.astype("int64")})
    os.makedirs(ht.TABLE_DIR, exist_ok=True)
    out.to_parquet(ht.LENGTHS_PATH, index=False, compression="zstd")
    import tokenizers
    import transformers
    rec = {"path": "data/processed/hotpot/lengths.parquet", "sha256": ht.file_sha256(ht.LENGTHS_PATH),
           "n_rows": int(len(out)), "raw_sha256": ht.FILE_SHA256, "length_models": list(ht.LENGTH_MODELS),
           "tokenizers": pins, "transformers": transformers.__version__, "tokenizers_version": tokenizers.__version__,
           "definition": __doc__.split("\n\n")[2].strip(), "runtime_s": round(time.time() - t0, 1)}
    ht.write_json(rec, os.path.join(ht.TABLE_DIR, "lengths.json"))
    q = out[["item_tokens", "demo_tokens", "gold_tokens"]].quantile([0.5, 0.9, 0.99, 1.0])
    print(q.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
