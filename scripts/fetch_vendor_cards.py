r"""Fetch the generation_config.json and model card (README.md) of the three looping native-reasoning checkpoints at their pinned
revisions (config/decoding_thinking_vendor.yaml records what they say).

From the repository root:
    .\.venv\Scripts\python.exe scripts\fetch_vendor_cards.py

Writes manifests/external/vendor_cards/<model_key>/{generation_config.json,README.md} and a record with each file's sha256.
Two small text files per model; no weights.
"""
import hashlib
import json
import os
import shutil
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "manifests", "external", "vendor_cards")
MODELS = ("qwen3.5-35b-a3b", "qwen3.6-35b-a3b", "qwen3.6-27b")
FILES = ("generation_config.json", "README.md")


def main():
    from huggingface_hub import hf_hub_download
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models_thinking_gsm8k.yaml"), encoding="utf-8"))
    rec = {}
    for k in MODELS:
        e = reg["dataset2"]["models"][k]
        d = os.path.join(OUT, k)
        os.makedirs(d, exist_ok=True)
        rec[k] = {"hf_id": e["hf_id"], "revision": e["revision"], "files": {}}
        for f in FILES:
            p = hf_hub_download(e["hf_id"], f, revision=e["revision"])
            dst = os.path.join(d, f)
            shutil.copyfile(p, dst)
            rec[k]["files"][f] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    with open(os.path.join(OUT, "record.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    sys.exit(main())
