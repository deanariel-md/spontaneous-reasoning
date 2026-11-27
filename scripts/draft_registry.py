r"""Draft Dataset 2 registry entries from a panel list, for review. Logic: src/registry_draft.py.

Reads a YAML list of entries (hf_id required; revision, family, tier, key optional), resolves each
revision to a 40-hex commit sha through the Hugging Face Hub metadata API (cached login; no weights are
downloaded and the only file fetched is config.json, into a temporary directory), and proposes
tensor_parallel_size, max_model_len, dtype and the thinking-off method for every model. It writes:

    config/models_draft.yaml               entries under dataset2.models in the registry schema, each
                                           preceded by a comment block with its facts and arithmetic
    config/models_draft_provenance.json    full facts per model (API fields, dtype breakdown, TP and
                                           length arithmetic, thinking-off source, validation)
    config/models_draft_summary.md         table of key, family, sha, weights GB, TP, max_model_len and
    config/models_draft_summary_clean.md   thinking-off method, with the problems listed

It never writes config/models.yaml. Every drafted entry is parsed with generation.parse_model_spec; a
PLACEHOLDER entry is an expected refusal and is listed. The draft is reviewed and its entries copied by hand.

Panel file example:

    - hf_id: Qwen/Qwen3-0.6B
      family: Qwen3
    - hf_id: ibm-granite/granite-3.3-2b-instruct
      revision: main
      key: granite-3.3-2b

Usage (PowerShell):
    .\.venv\Scripts\python.exe scripts\draft_registry.py --panel panel.yaml
    .\.venv\Scripts\python.exe scripts\draft_registry.py --panel panel.yaml --out config\models_draft.yaml `
        --gpu-mem-gb 80 --prompt-budget 512 --max-tp 8 --overwrite

Exit codes: 0 written and every model resolved, 3 written with at least one model not resolved (listed in
the summary), 2 refused (bad panel file or options, --out is config/models.yaml, outputs exist without
--overwrite), 1 any other error.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import registry_draft  # noqa: E402

if __name__ == "__main__":
    sys.exit(registry_draft.main())
