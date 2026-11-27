# Credits

This code uses public benchmarks, tools, a capability index and many language models. Each is listed with its source and licence. Citations follow the paper's reference list.

## Benchmarks

| Benchmark | Source | Licence | Citation |
|---|---|---|---|
| GSM8K | https://github.com/openai/grade-school-math (Hugging Face `openai/gsm8k`) | MIT | Cobbe, K. et al. Training verifiers to solve math word problems. arXiv:2110.14168 (2021). |
| MATH (training items for few-shot demonstrations and screening) | https://github.com/hendrycks/math (Hugging Face `EleutherAI/hendrycks_math`) | MIT | Hendrycks, D. et al. Measuring mathematical problem solving with the MATH dataset. NeurIPS Datasets and Benchmarks Track (2021). |
| MATH-500 | https://github.com/openai/prm800k (Hugging Face `HuggingFaceH4/MATH-500`) | MIT | Lightman, H. et al. Let's verify step by step. ICLR (2024). |
| BIG-Bench Hard | https://github.com/suzgunmirac/BIG-Bench-Hard | MIT | Suzgun, M. et al. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. Findings of ACL 2023, 13003-13051. |
| HotpotQA (distractor setting) | https://hotpotqa.github.io (Hugging Face `hotpotqa/hotpot_qa`) | CC BY-SA 4.0 | Yang, Z. et al. HotpotQA: a dataset for diverse, explainable multi-hop question answering. EMNLP (2018). |
| DOVE (DOVE_Lite, used by the pilot tool's first example and by the DOVE settings of four analyses) | https://huggingface.co/datasets/nlphuji/DOVE_Lite | CDLA-Permissive-2.0 | Habba, E. et al. DOVE: a large-scale multi-dimensional predictions dataset towards meaningful LLM evaluation. Findings of ACL 2025, 11744-11763. |

## Tools and data whose code or outputs are used

- PromptSuite 3.0.7 generated the candidate prompt templates. Habba, E., Dahan, N., Lior, G. and Stanovsky, G. PromptSuite: a task-agnostic framework for multi-prompt generation. EMNLP 2025 System Demonstrations, 254-263. https://github.com/eliyahabba/PromptSuite. Licence: MIT License (the package metadata of PromptSuite 3.0.7).
- Epoch Capabilities Index, Epoch AI, CC BY 4.0, retrieved 30 October 2026, https://epoch.ai/eci. It chose the model panel (`manifests/external/eci_scores_20261030.csv`) and gives the capability scores in `pilot_report/examples/gsm8k29_capability.csv`. Cited as Epoch AI. Epoch Capabilities Index (2025).
- vLLM 0.30.0 (https://github.com/vllm-project/vllm, Apache License 2.0) ran the local models; PyTorch 2.13.0 (BSD-3-Clause) underneath.
- Hugging Face `transformers`, `tokenizers` and `huggingface_hub` (Apache License 2.0) render chat templates and count tokens; `tiktoken` (MIT) counts tokens for the API models.
- NumPy, pandas, SciPy, PyYAML, PyArrow and pytest run the analyses and tests, each under its own open-source licence.

## Open models

The panel is every eligible checkpoint of five families. Licence names are those on each model card, as recorded from the Hugging Face model metadata on 30 October 2026 (`manifests/d2_panel_extension.json`).

| Model (Hugging Face id) | Licence |
|---|---|
| google/gemma-2b-it | Gemma Terms of Use |
| google/gemma-7b-it | Gemma Terms of Use |
| google/gemma-3-4b-it | Gemma Terms of Use |
| google/gemma-2-9b-it | Gemma Terms of Use |
| google/gemma-2-27b-it | Gemma Terms of Use |
| google/gemma-3-12b-it | Gemma Terms of Use |
| google/gemma-3-27b-it | Gemma Terms of Use |
| google/gemma-4-26B-A4B-it | Apache License 2.0 |
| google/gemma-4-31B-it | Apache License 2.0 |
| meta-llama/Llama-2-7b-chat-hf | Llama 2 Community License Agreement |
| meta-llama/Llama-2-13b-chat-hf | Llama 2 Community License Agreement |
| meta-llama/Meta-Llama-3-8B-Instruct | Meta Llama 3 Community License Agreement |
| meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 Community License Agreement |
| mistralai/Mistral-7B-Instruct-v0.1 | Apache License 2.0 |
| mistralai/Mistral-7B-Instruct-v0.3 | Apache License 2.0 |
| mistralai/Mistral-Nemo-Instruct-2407 | Apache License 2.0 |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | Apache License 2.0 |
| microsoft/Phi-3-mini-4k-instruct | MIT License |
| microsoft/Phi-3-medium-4k-instruct | MIT License |
| microsoft/phi-4 | MIT License |
| Qwen/Qwen2.5-7B-Instruct | Apache License 2.0 |
| Qwen/Qwen3-8B | Apache License 2.0 |
| Qwen/Qwen3-14B | Apache License 2.0 |
| Qwen/Qwen3-32B | Apache License 2.0 |
| Qwen/Qwen3-30B-A3B | Apache License 2.0 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | Apache License 2.0 |
| Qwen/Qwen3.5-9B | Apache License 2.0 |
| Qwen/Qwen3.5-35B-A3B | Apache License 2.0 |
| Qwen/Qwen3.6-27B | Apache License 2.0 |
| Qwen/Qwen3.6-35B-A3B | Apache License 2.0 |

The six larger models:

| Model | Run as | Licence | Citation |
|---|---|---|---|
| Ministral 3 8B Instruct 2512 | mistralai/Ministral-3-8B-Instruct-2512-BF16, local | Apache License 2.0 | Mistral AI. Introducing Mistral 3 (2025). |
| Ministral 3 14B Instruct 2512 | mistralai/Ministral-3-14B-Instruct-2512-BF16, local | Apache License 2.0 | Mistral AI. Introducing Mistral 3 (2025). |
| Mistral Large 3 (2512) | Mistral API | Apache License 2.0 | Mistral AI. Mistral-Large-3-675B-Instruct-2512, model card (2025). |
| Llama-3.1-70B-Instruct | RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8-dynamic, local | Llama 3.1 Community License Agreement | |
| Qwen3.5-122B-A10B | Qwen/Qwen3.5-122B-A10B-FP8, local | Apache License 2.0 | Qwen Team. Qwen3 technical report. arXiv:2505.09388 (2025). |
| DeepSeek V4.1-Flash-1017 | Nebius API; open weights deepseek-ai/DeepSeek-V4.1-Flash-1017 | MIT License | DeepSeek-AI. DeepSeek-V4.1-Flash: smarter, faster, more efficient (2026). |

The code also analyses the outputs of two earlier collections, which it reads but does not run: the DOVE setting (DOVE_Lite's MMLU block, `manifests/dove_panel.json`) and the reproduction of Cheng et al. on GSM8K (`config/models_reproduction.yaml`). Their models not listed above, with the licence named in each model card's Hugging Face metadata (read 27 November 2026):

| Model (Hugging Face id) | Used in | Licence |
|---|---|---|
| allenai/OLMoE-1B-7B-0924-Instruct | DOVE setting | Apache License 2.0 |
| meta-llama/Llama-3.2-1B-Instruct | DOVE setting; reproduction | Llama 3.2 Community License Agreement |
| meta-llama/Llama-3.2-3B-Instruct | DOVE setting; reproduction | Llama 3.2 Community License Agreement |
| Qwen/Qwen2.5-0.5B-Instruct | reproduction | Apache License 2.0 |
| Qwen/Qwen2.5-1.5B-Instruct | reproduction | Apache License 2.0 |
| Qwen/Qwen2.5-14B-Instruct | reproduction | Apache License 2.0 |
| Qwen/Qwen2.5-32B-Instruct | reproduction | Apache License 2.0 |
| RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic | reproduction | Qwen License Agreement (named in the card of Qwen/Qwen2.5-72B-Instruct; the FP8 copy's metadata names none) |
| google/gemma-2-2b-it | reproduction | Gemma Terms of Use |
| mistralai/Ministral-8B-Instruct-2410 | reproduction | Mistral AI Research License |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | reproduction | Llama 3.3 Community License Agreement |

The DOVE setting's other models (meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3) and the reproduction's others (Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it) are in the panel table above.

The template banks' instruction paraphrases were written by Llama 3.3 70B Instruct (RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic; licence: Llama 3.3 Community License Agreement).

## Commercial model

Qwen4-Max is a commercial model of Alibaba Cloud's Qwen team. It was reached through OpenRouter (https://openrouter.ai) with the provider fixed to Alibaba.

The reading audit's judge is Claude Opus 5 (`claude-opus-5`), a commercial model of Anthropic. It was reached through the Anthropic Message Batches API (https://api.anthropic.com). `scripts/reading_audit.py` needs its own API key to run the judge step.

## Model providers

- Anthropic (https://www.anthropic.com): Claude Opus 5, the reading audit's judge.
- OpenRouter (https://openrouter.ai): Qwen4-Max.
- Mistral AI (https://mistral.ai): Mistral Large 3.
- Nebius AI Studio (https://nebius.com): DeepSeek V4.1-Flash-1017.
