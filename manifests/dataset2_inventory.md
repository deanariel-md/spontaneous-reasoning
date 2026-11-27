# Dataset 2 inventory: checkpoints with a controllable thinking switch

Inventory date: 3 October 2026. No intervention is recommended in this file.

## Conventions

- Window: Hugging Face repository created on or before 30 June 2026. The release column is the repository creation date (UTC); vendor announcements can differ by a few days.
- Size envelope: dense checkpoints up to 72.7B total parameters; MoE checkpoints whose weights are at most 320 GB (4 x 80 GB) at the released precision or at BF16. Rows outside the envelope are in Table 1b and are excluded from the counts.
- Total parameters are the safetensors totals reported by the Hugging Face API. They include embeddings and any vision encoder or multi-token-prediction head. Active parameters come from the model card; "(name)" marks a value taken from the checkpoint name only.
- Weights GB is computed from the safetensors dtypes at the released precision.
- Evidence class. C: the checkpoint's model card (or a repository file the card cites) describes the switch. O: vendor documentation outside the card describes it. T: the flag is visible only in the chat template or in sample code.
- Off-mode code. P: the chat template inserts an empty reasoning block when the switch is off (template text read). C: the card describes a non-reasoning mode. B: the card states the model is instructed to answer directly at budget 0. O: reasoning is opt-in and off is the default behavior of the instruct model.
- Reading method: model cards, chat templates and metadata were read from huggingface.co without authentication on 2026-10-03. Gated repositories (Llama, Gemma 3, Command A Reasoning) expose metadata only. No weights or datasets were downloaded and no model was run.
- In the cards read, no statement says that off-mode visible output is free of step-by-step text. Off means that the model does not emit its separate reasoning block.

## Table 1. Documented thinking switch, inside the size envelope

Mechanisms:

| Key | Lines | Exact control |
|---|---|---|
| M1 | Qwen3 | `enable_thinking=True` or `enable_thinking=False` passed to `apply_chat_template` (or `chat_template_kwargs` in vLLM/SGLang). Soft switch: `/think` and `/no_think` in a user or system message, honored when `enable_thinking=True`. |
| M2 | Qwen3.5, Qwen3.6 | `enable_thinking` through `chat_template_kwargs` (or `apply_chat_template`). The cards state that the Qwen3 soft switch (`/think`, `/nothink`) is not officially supported. |
| M3 | Gemma 4 | Thinking is on when the `<\|think\|>` token starts the system prompt; the chat template exposes this as `enable_thinking` (template default false). |
| M4 | Llama-Nemotron v1 | System prompt `detailed thinking on` or `detailed thinking off`; instructions go in the user turn. |
| M5 | Nemotron v1.5 / Nano v2 / Cascade | `/think` or `/no_think` in the system prompt (Super-49B-v1.5, Nano-9B-v2, Nano-12B-v2; Nano v2 also accepts the tags in user messages). Cascade-8B appends the tag to the user input and also takes `enable_thinking`. |
| M6 | Nemotron 3 | `enable_thinking=False` through `chat_template_kwargs` or `apply_chat_template` (default True). Optional `low_effort` and `reasoning_budget` on the Nano, Super and Puzzle cards. |
| M7 | GLM | `enable_thinking=False` through `chat_template_kwargs` (the GLM-4.5 card documents this for vLLM and SGLang; the template appends `/nothink` or an empty think block). Z.ai API: `thinking` with `type` set to `enabled` or `disabled`. |
| M8 | Mistral | `reasoning_effort` set to `none` or `high` per request (the chat template raises an error for other values; undefined means `none`). |
| M9 | SmolLM3 | `/think` or `/no_think` in the system prompt, or the `enable_thinking` template kwarg. |
| M10 | Hunyuan | `enable_thinking=False` in `apply_chat_template`, or `/no_think` and `/think` placed before the prompt. |
| M11 | Seed-OSS | `thinking_budget` in tokens. Default is unlimited thinking. The card states that the model is instructed to output a direct response when the budget is 0. |
| M12 | EXAONE | `enable_thinking=True` in `apply_chat_template`. EXAONE 4.0 default is non-reasoning; EXAONE 4.5 default is reasoning (`enable_thinking=False` for non-reasoning). |
| M13 | Hermes 4 | `thinking=True` chat-template flag, or a documented deep-thinking system prompt. Reasoning is opt-in. |
| M14 | Cogito | `enable_thinking=True` in the chat template, or system prompt `Enable deep thinking subroutine.` Reasoning is opt-in. |
| M15 | MiniCPM | `enable_thinking=True` or `enable_thinking=False` in `apply_chat_template`. |
| M16 | Intern-S1-mini, Intern-S2-Preview | `enable_thinking=False` in `apply_chat_template` or `chat_template_kwargs` (thinking on by default). The S2-Preview card does not recommend disabling thinking for agentic tasks. |
| M17 | InternLM3 | A long documented thinking system prompt (math-competition persona) selects the thinking mode; the default is the normal response mode. No template flag. |
| M18 | Granite 3.x | Boolean `thinking=True` or `thinking=False` argument to `apply_chat_template` (the cards show outputs for both). |
| M19 | DeepSeek-V4 | Messages are encoded with the repo's `encoding_dsv4.encode_messages` using `thinking_mode` set to `chat` or `thinking`; no Jinja template ships. The card lists three modes: Non-think, Think High, Think Max. |
| M20 | Sarvam | `enable_thinking` in `apply_chat_template`. Sarvam-M card: hybrid think and non-think modes. Sarvam-30B and 105B: the flag appears in sample code, and the template contains a `<\|nothink\|>` branch. |
| M21 | HyperCLOVA X SEED Think | Think-14B: assistant prefix `assistant/think` for reasoning and `assistant` for a direct answer in the ChatML prompt. Think-32B: `chat_template_kwargs` with `thinking` set to True, plus optional `thinking_token_budget`. |

Checkpoints:

| Vendor | Line | Checkpoint | Params total (B) / active (B) | Weights GB | Released | Switch | Default | Off | Evid. | License | Notes | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alibaba | Qwen3 | `Qwen/Qwen3-0.6B` | 0.75 dense | 2 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-0.6B |
| Alibaba | Qwen3 | `Qwen/Qwen3-1.7B` | 2.03 dense | 4 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-1.7B |
| Alibaba | Qwen3 | `Qwen/Qwen3-4B` | 4.02 dense | 8 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-4B |
| Alibaba | Qwen3 | `Qwen/Qwen3-8B` | 8.19 dense | 16 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-8B |
| Alibaba | Qwen3 | `Qwen/Qwen3-14B` | 14.8 dense | 30 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-14B |
| Alibaba | Qwen3 | `Qwen/Qwen3-32B` | 32.8 dense | 66 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-32B |
| Alibaba | Qwen3 | `Qwen/Qwen3-30B-A3B` | 30.5 / 3.3 | 61 | 2025-04-27 | M1 | on | P | C | Apache-2.0 |  | https://huggingface.co/Qwen/Qwen3-30B-A3B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-0.8B` | 0.87 dense | 2 | 2026-02-28 | M2 | off | P | C | Apache-2.0 | Vision-language checkpoint; parameter count includes the vision encoder. | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-2B` | 2.27 dense | 5 | 2026-02-28 | M2 | off | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-2B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-4B` | 4.66 dense | 9 | 2026-02-27 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-4B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-9B` | 9.65 dense | 19 | 2026-02-27 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-9B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-27B` | 27.8 dense | 56 | 2026-02-24 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-27B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-35B-A3B` | 36.0 / 3 | 72 | 2026-02-24 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |
| Alibaba | Qwen3.5 | `Qwen/Qwen3.5-122B-A10B` | 125.1 / 10 | 250 | 2026-02-24 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.5-122B-A10B |
| Alibaba | Qwen3.6 | `Qwen/Qwen3.6-27B` | 27.8 dense | 56 | 2026-04-21 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.6-27B |
| Alibaba | Qwen3.6 | `Qwen/Qwen3.6-35B-A3B` | 36.0 / 3 | 72 | 2026-04-15 | M2 | on | P | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/Qwen/Qwen3.6-35B-A3B |
| Google | Gemma 4 | `google/gemma-4-E2B-it` | 5.12 / 2.3 effective | 10 | 2026-03-02 | M3 | off | C | C | Apache-2.0 | Card: the empty-thought-block behavior for thinking off is stated for all Gemma 4 models except E2B and E4B. Multimodal. | https://huggingface.co/google/gemma-4-E2B-it |
| Google | Gemma 4 | `google/gemma-4-E4B-it` | 8.00 / 4.5 effective | 16 | 2026-03-02 | M3 | off | C | C | Apache-2.0 | Same off-mode note as E2B. Multimodal. | https://huggingface.co/google/gemma-4-E4B-it |
| Google | Gemma 4 | `google/gemma-4-12B-it` | 12.0 dense | 24 | 2026-05-23 | M3 | off | P | C | Apache-2.0 | Template inserts an empty thought channel when off. | https://huggingface.co/google/gemma-4-12B-it |
| Google | Gemma 4 | `google/gemma-4-26B-A4B-it` | 25.8 / 3.8 | 52 | 2026-03-11 | M3 | off | P | C | Apache-2.0 | Template inserts an empty thought channel when off. | https://huggingface.co/google/gemma-4-26B-A4B-it |
| Google | Gemma 4 | `google/gemma-4-31B-it` | 31.3 dense | 63 | 2026-03-11 | M3 | off | P | C | Apache-2.0 | Template inserts an empty thought channel when off. | https://huggingface.co/google/gemma-4-31B-it |
| NVIDIA | Llama-Nemotron | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | 4.51 dense | 9 | 2025-05-03 | M4 | set by system prompt | C | C | NVIDIA Open Model License + Llama 3.1 Community License | Derived from Llama-3.1-Minitron-4B-Width-Base. | https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 |
| NVIDIA | Llama-Nemotron | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | 8.03 dense | 16 | 2025-03-16 | M4 | set by system prompt | C | C | NVIDIA Open Model License + Llama 3.1 Community License | Derived from Llama-3.1-8B-Instruct. | https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1 |
| NVIDIA | Llama-Nemotron | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | 49.9 dense | 100 | 2025-03-16 | M4 | set by system prompt | C | C | NVIDIA Open Model License + Llama 3.3 Community License | Derived from Llama-3.3-70B-Instruct. | https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1 |
| NVIDIA | Llama-Nemotron | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 49.9 dense | 100 | 2025-07-25 | M5 | on (empty system prompt) | C | C | NVIDIA Open Model License + Llama 3.3 Community License | Card recommends greedy decoding for reasoning off. | https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 |
| NVIDIA | Nemotron Nano 2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 8.89 dense | 18 | 2025-08-12 | M5 | on | P | C | NVIDIA Open Model License |  | https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2 |
| NVIDIA | Nemotron Nano 2 | `nvidia/NVIDIA-Nemotron-Nano-12B-v2` | 12.3 dense | 25 | 2025-08-21 | M5 | on | C | C | NVIDIA Open Model License |  | https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2 |
| NVIDIA | Nemotron 3 | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 3.97 dense | 8 | 2026-03-07 | M6 | on | C | C | NVIDIA Nemotron Open Model License |  | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 |
| NVIDIA | Nemotron 3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 31.6 / 3.5 | 63 | 2025-12-04 | M6 | on | C | C | NVIDIA Nemotron Open Model License | Card recommends greedy search for reasoning off. | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| NVIDIA | Nemotron 3 | `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16` | 31.6 / ~3 (name) | 63 | 2026-04-01 | M6 | on | C | C | NVIDIA Open Model License | Nested elastic sub-models; the card states elastic budget control is not supported in standard vLLM. | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 |
| NVIDIA | Nemotron 3 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 123.6 / 12 | 247 | 2026-03-10 | M6 | on | C | C | NVIDIA Nemotron Open Model License |  | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 |
| NVIDIA | Nemotron 3 | `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16` | 75.4 / 9 (name) | 157 | 2026-06-24 | M6 | on | C | C | OpenMDW-1.1 | Compressed derivative of Nemotron-3-Super-120B-A12B. | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 |
| NVIDIA | Nemotron-Cascade | `nvidia/Nemotron-Cascade-8B` | 8.19 dense | 16 | 2025-12-16 | M5 | set by user-input tag | C | C | NVIDIA Open Model License | Post-trained from Qwen3-8B-Base; the card lists thinking and instruct modes. | https://huggingface.co/nvidia/Nemotron-Cascade-8B |
| NVIDIA | Nemotron-Cascade | `nvidia/Nemotron-Cascade-2-30B-A3B` | 31.6 / 3 | 63 | 2026-03-18 | M6 | on | P | C | NVIDIA Open Model License | Post-trained from Nemotron-3-Nano-30B-A3B-Base. Card: instruct mode prepends `<think></think>` to the assistant turn. | https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B |
| Z.ai | GLM | `zai-org/GLM-4.5-Air` | 110.5 / 12 | 221 | 2025-07-20 | M7 | on | P | C | MIT | Card describes thinking and non-thinking modes; the flag is documented on the GLM-4.5 card and read in this checkpoint's template. | https://huggingface.co/zai-org/GLM-4.5-Air |
| Z.ai | GLM | `zai-org/GLM-4.7-Flash` | 31.2 / ~3 (card: 30B-A3B) | 62 | 2026-01-19 | M7 | on | P | T | MIT | Template has `enable_thinking`; docs.z.ai documents `thinking.type` for the GLM-4.7 series, and the fetched text does not list Flash separately. | https://huggingface.co/zai-org/GLM-4.7-Flash |
| Mistral AI | Mistral Small 4 | `mistralai/Mistral-Small-4-119B-2603` | 119.4 / 6.5 | 121 | 2026-01-23 | M8 | off (`none`) | C | C | Apache-2.0 | Card: `none` matches Mistral-Small-3.2 chat style. Most weights are released in FP8. | https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 |
| Hugging Face | SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | 3.08 dense | 6 | 2025-07-08 | M9 | on | C | C | Apache-2.0 |  | https://huggingface.co/HuggingFaceTB/SmolLM3-3B |
| Tencent | Hunyuan | `tencent/Hunyuan-0.5B-Instruct` | 0.54 dense | 1 | 2025-07-30 | M10 | on (slow thinking) | C | C | Tencent Hunyuan Community License (excludes EU, UK, South Korea; text read) |  | https://huggingface.co/tencent/Hunyuan-0.5B-Instruct |
| Tencent | Hunyuan | `tencent/Hunyuan-1.8B-Instruct` | 1.79 dense | 4 | 2025-07-30 | M10 | on (slow thinking) | C | C | Tencent Hunyuan Community License (excludes EU, UK, South Korea; text read) |  | https://huggingface.co/tencent/Hunyuan-1.8B-Instruct |
| Tencent | Hunyuan | `tencent/Hunyuan-4B-Instruct` | 4.22 dense | 8 | 2025-07-30 | M10 | on (slow thinking) | C | C | Tencent Hunyuan Community License (excludes EU, UK, South Korea; text read) |  | https://huggingface.co/tencent/Hunyuan-4B-Instruct |
| Tencent | Hunyuan | `tencent/Hunyuan-7B-Instruct` | 7.50 dense | 15 | 2025-07-30 | M10 | on (slow thinking) | C | C | Tencent Hunyuan Community License (excludes EU, UK, South Korea; text read) |  | https://huggingface.co/tencent/Hunyuan-7B-Instruct |
| Tencent | Hunyuan | `tencent/Hunyuan-A13B-Instruct` | 80.4 / 13 | 161 | 2025-06-25 | M10 | on (slow thinking) | C | C | Tencent Hunyuan Community License (excludes EU, UK, South Korea; text read) |  | https://huggingface.co/tencent/Hunyuan-A13B-Instruct |
| ByteDance Seed | Seed-OSS | `ByteDance-Seed/Seed-OSS-36B-Instruct` | 36.2 dense | 72 | 2025-08-20 | M11 | unlimited budget | B | C | Apache-2.0 |  | https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct |
| LG AI Research | EXAONE | `LGAI-EXAONE/EXAONE-4.0-1.2B` | 1.28 dense | 3 | 2025-07-11 | M12 | off | O | C | EXAONE AI Model License 1.2-NC (research and educational use; text read) |  | https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B |
| LG AI Research | EXAONE | `LGAI-EXAONE/EXAONE-4.0-32B` | 32.0 dense | 64 | 2025-07-11 | M12 | off | O | C | EXAONE AI Model License 1.2-NC (research and educational use; text read) |  | https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B |
| LG AI Research | EXAONE | `LGAI-EXAONE/EXAONE-4.0.1-32B` | 32.0 dense | 64 | 2025-07-29 | M12 | off | O | C | EXAONE AI Model License 1.2-NC (research and educational use; text read) |  | https://huggingface.co/LGAI-EXAONE/EXAONE-4.0.1-32B |
| LG AI Research | EXAONE | `LGAI-EXAONE/EXAONE-4.5-33B` | 34.4 dense | 69 | 2026-04-04 | M12 | on | C | C | EXAONE AI Model License 1.2-NC (research and educational use; text read) | Vision-language checkpoint. | https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B |
| Nous Research | Hermes 4 | `NousResearch/Hermes-4-14B` | 14.8 dense | 30 | 2025-08-30 | M13 | off | O | C | Apache-2.0 | Fine-tune of Qwen3-14B. | https://huggingface.co/NousResearch/Hermes-4-14B |
| Nous Research | Hermes 4 | `NousResearch/Hermes-4-70B` | 70.6 dense | 141 | 2025-08-18 | M13 | off | O | C | Llama 3 license | Fine-tune of Llama-3.1-70B. | https://huggingface.co/NousResearch/Hermes-4-70B |
| Nous Research | Hermes 4 | `NousResearch/Hermes-4.3-36B` | 36.2 dense | 72 | 2025-11-17 | M13 | off | O | C | Apache-2.0 | Fine-tune of Seed-OSS-36B. | https://huggingface.co/NousResearch/Hermes-4.3-36B |
| Deep Cogito | Cogito v1 preview | `deepcogito/cogito-v1-preview-llama-3B` | 3.61 dense | 7 | 2025-03-31 | M14 | off | C | C | Llama 3.2 Community License |  | https://huggingface.co/deepcogito/cogito-v1-preview-llama-3B |
| Deep Cogito | Cogito v1 preview | `deepcogito/cogito-v1-preview-llama-8B` | 8.03 dense | 16 | 2025-03-31 | M14 | off | C | C | Llama 3.1 Community License |  | https://huggingface.co/deepcogito/cogito-v1-preview-llama-8B |
| Deep Cogito | Cogito v1 preview | `deepcogito/cogito-v1-preview-qwen-14B` | 14.8 dense | 30 | 2025-03-31 | M14 | off | C | C | Apache-2.0 |  | https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B |
| Deep Cogito | Cogito v1 preview | `deepcogito/cogito-v1-preview-qwen-32B` | 32.8 dense | 66 | 2025-03-31 | M14 | off | C | C | Apache-2.0 |  | https://huggingface.co/deepcogito/cogito-v1-preview-qwen-32B |
| Deep Cogito | Cogito v1 preview | `deepcogito/cogito-v1-preview-llama-70B` | 70.6 dense | 141 | 2025-03-31 | M14 | off | C | C | Llama 3.1 Community License |  | https://huggingface.co/deepcogito/cogito-v1-preview-llama-70B |
| Deep Cogito | Cogito v2 preview | `deepcogito/cogito-v2-preview-llama-70B` | 70.6 dense | 141 | 2025-07-27 | M14 | off | C | C | Llama 3.1 Community License | Card: reasoning-on output is started with `<think>`. | https://huggingface.co/deepcogito/cogito-v2-preview-llama-70B |
| Deep Cogito | Cogito v2 preview | `deepcogito/cogito-v2-preview-llama-109B-MoE` | 108.6 / 17 (Llama-4-Scout base) | 217 | 2025-07-27 | M14 | off | C | C | Llama 4 Community License |  | https://huggingface.co/deepcogito/cogito-v2-preview-llama-109B-MoE |
| OpenBMB | MiniCPM | `openbmb/MiniCPM4.1-8B` | 8.19 dense | 16 | 2025-09-02 | M15 | model-decided unless set off | P | C | Apache-2.0 | Template inserts an empty think block only when `enable_thinking` is false. | https://huggingface.co/openbmb/MiniCPM4.1-8B |
| OpenBMB | MiniCPM | `openbmb/MiniCPM5-1B` | 1.08 dense | 2 | 2026-05-21 | M15 | per kwarg | C | C | Apache-2.0 |  | https://huggingface.co/openbmb/MiniCPM5-1B |
| Shanghai AI Lab | InternLM3 / Intern-S | `internlm/internlm3-8b-instruct` | 8.80 dense | 18 | 2025-01-13 | M17 | off (normal mode) | O | C | Apache-2.0 | Thinking mode is a long system prompt. | https://huggingface.co/internlm/internlm3-8b-instruct |
| Shanghai AI Lab | InternLM3 / Intern-S | `internlm/Intern-S1-mini` | 8.54 dense | 17 | 2025-08-18 | M16 | on | C | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/internlm/Intern-S1-mini |
| Shanghai AI Lab | InternLM3 / Intern-S | `internlm/Intern-S2-Preview` | 36.1 dense | 73 | 2026-05-15 | M16 | on | C | C | Apache-2.0 | Vision-language checkpoint. | https://huggingface.co/internlm/Intern-S2-Preview |
| IBM | Granite 3.x | `ibm-granite/granite-3.2-2b-instruct` | 2.53 dense | 5 | 2025-02-17 | M18 | per kwarg | C | C | Apache-2.0 |  | https://huggingface.co/ibm-granite/granite-3.2-2b-instruct |
| IBM | Granite 3.x | `ibm-granite/granite-3.2-8b-instruct` | 8.17 dense | 16 | 2025-02-17 | M18 | per kwarg | C | C | Apache-2.0 |  | https://huggingface.co/ibm-granite/granite-3.2-8b-instruct |
| IBM | Granite 3.x | `ibm-granite/granite-3.3-2b-instruct` | 2.53 dense | 5 | 2025-04-09 | M18 | per kwarg | C | C | Apache-2.0 |  | https://huggingface.co/ibm-granite/granite-3.3-2b-instruct |
| IBM | Granite 3.x | `ibm-granite/granite-3.3-8b-instruct` | 8.17 dense | 16 | 2025-04-09 | M18 | per kwarg | C | C | Apache-2.0 |  | https://huggingface.co/ibm-granite/granite-3.3-8b-instruct |
| DeepSeek | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash` | 290.9 / 13 | 160 | 2026-04-22 | M19 | chosen at encoding | C | C | MIT | 284B total per card; FP4 experts plus FP8 weights; HF repo about 160 GB. | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| Sarvam AI | Sarvam | `sarvamai/sarvam-m` | 23.6 dense | 47 | 2025-05-20 | M20 | on | C | C | Apache-2.0 | Mistral architecture, 23.6B. | https://huggingface.co/sarvamai/sarvam-m |
| Sarvam AI | Sarvam | `sarvamai/sarvam-30b` | 32.2 / 2.4 (non-embedding) | 129 | 2026-03-03 | M20 | per kwarg | C | T | Apache-2.0 | Released in FP32; BF16 footprint about 64 GB. | https://huggingface.co/sarvamai/sarvam-30b |
| Sarvam AI | Sarvam | `sarvamai/sarvam-105b` | 106.0 / 10.3 | 424 | 2026-03-03 | M20 | per kwarg | C | T | Apache-2.0 | Released in FP32 (about 424 GB); BF16 footprint about 212 GB. | https://huggingface.co/sarvamai/sarvam-105b |
| NAVER | HyperCLOVA X SEED Think | `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` | 14.7 dense | 59 | 2025-07-21 | M21 | per prompt prefix | C | C | HyperCLOVA X SEED Think license (custom; text not read) |  | https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B |
| NAVER | HyperCLOVA X SEED Think | `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | 33.3 dense | 67 | 2025-12-23 | M21 | per kwarg | C | C | HyperCLOVA X SEED license (custom; text not read) | Vision-language checkpoint. | https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B |

## Table 1b. Documented thinking switch, outside the size envelope

| Checkpoint | Vendor | Active (B) | Why outside | Switch | Evid. | License | Source |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-235B-A22B` | Alibaba | 22 | MoE, 470 GB at BF16; the official FP8 repo `Qwen/Qwen3-235B-A22B-FP8` is about 239 GB | See M1 in Table 1. | C | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-235B-A22B |
| `Qwen/Qwen3.5-397B-A17B` | Alibaba | 17 | MoE, about 807 GB at BF16; the official FP8 repo is about 400 GB | See M2 in Table 1. | C | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-397B-A17B |
| `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | NVIDIA | dense | Dense 253B | See M4 in Table 1. | C | NVIDIA Open Model License + Llama 3.1 Community License | https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 |
| `zai-org/GLM-4.5` | Z.ai | 32 | MoE 358B, 717 GB at BF16 | See M7 in Table 1. | C | MIT | https://huggingface.co/zai-org/GLM-4.5 |
| `zai-org/GLM-4.7` | Z.ai | unverified | MoE 358B, 717 GB at BF16 | docs.z.ai: thinking is on by default in the GLM-4.7 series; Z.ai API `thinking` with `type` set to `disabled`. Template flag not read | O | MIT | https://huggingface.co/zai-org/GLM-4.7 |
| `deepseek-ai/DeepSeek-V3.1` | DeepSeek | 37 (DeepSeek-V4 card table) | MoE 685B | Card flag `thinking=True` in the chat template | C | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 |
| `tencent/Hy3-preview` | Tencent | 21 | MoE 295B, 598 GB at BF16 | `reasoning_effort` set to `no_think` (default), `low` or `high` through `chat_template_kwargs` | C | Custom (HF tag: other; text not read) | https://huggingface.co/tencent/Hy3-preview |
| `MiniMaxAI/MiniMax-M3` | MiniMax | unverified | MoE 427B | `thinking` parameter set to `enabled`, `adaptive` or `disabled` (card) | C | minimax-community (text not read) | https://huggingface.co/MiniMaxAI/MiniMax-M3 |
| `mistralai/Mistral-Medium-3.5-128B` | Mistral AI | dense | Dense 127.7B (about 134 GB, mostly FP8) | See M8 in Table 1. | C | Modified MIT License (exceptions for companies with large revenue) | https://huggingface.co/mistralai/Mistral-Medium-3.5-128B |
| `CohereLabs/command-a-reasoning-08-2025` | Cohere | dense | Dense 111B (222 GB at BF16); gated, card unread | docs.cohere.com: `thinking={"type": "disabled"}` (on by default); local flag unverified | O | CC-BY-NC-4.0 | https://huggingface.co/CohereLabs/command-a-reasoning-08-2025 |
| `CohereLabs/command-a-plus-05-2026-bf16` | Cohere | 25 | MoE 218.8B, 437 GB at BF16 (a 4-bit sibling repo exists) | Chat template: `reasoning` boolean or `reasoning_effort="none"`; the card shows thinking output and no off instruction | T | Apache-2.0 | https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16 |

Releases after the window that expose a switch (not counted): `Qwen/Qwen3.8-27B` (2026-08-05; `enable_thinking` and `reasoning_effort`), `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (2026-08-01; `enable_thinking`), `ibm-granite/granite-4.2-3b`, `-8b` and `-30b` (2026-08-07; `enable_thinking` and `low_effort`; the 8B card was read). Not examined for a switch: `swiss-ai/Apertus-v1.5-8B` and `-70B` (2026-07-24, gated) and `deepseek-ai/DeepSeek-V4.1-Flash` (2026-09-10).

## Table 2. Instruct checkpoints without a thinking switch

Direct-answer column: "Yes" means the checkpoint is a non-reasoning instruct model, so a direct-answer condition exists by default. "Yes by model type" marks gated repositories whose card text could not be read. The rows list widely used families and are not exhaustive.

| Vendor | Checkpoint | Params total (B) / active (B) | Weights GB | Released | Direct-answer condition | License | Source |
|---|---|---|---|---|---|---|---|
| Alibaba | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49 dense | 1 | 2024-09-16 | Yes. Non-reasoning instruct; the card has no reasoning mode. | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| Alibaba | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54 dense | 3 | 2024-09-17 | Yes. Same. | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| Alibaba | `Qwen/Qwen2.5-3B-Instruct` | 3.09 dense | 6 | 2024-09-17 | Yes. Same. | Qwen Research license | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct |
| Alibaba | `Qwen/Qwen2.5-7B-Instruct` | 7.62 dense | 15 | 2024-09-16 | Yes. Same. | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Alibaba | `Qwen/Qwen2.5-14B-Instruct` | 14.8 dense | 30 | 2024-09-16 | Yes. Same. | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct |
| Alibaba | `Qwen/Qwen2.5-32B-Instruct` | 32.8 dense | 66 | 2024-09-17 | Yes. Same. | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct |
| Alibaba | `Qwen/Qwen2.5-72B-Instruct` | 72.7 dense | 145 | 2024-09-16 | Yes. Same. | Qwen license | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct |
| Alibaba | `Qwen/Qwen3-4B-Instruct-2507` | 4.02 dense | 8 | 2025-08-05 | Yes. Card: supports only non-thinking mode. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| Alibaba | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30.5 / 3.3 | 61 | 2025-07-28 | Yes. Card: supports only non-thinking mode. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 |
| Alibaba | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 81.3 / ~3 (name) | 163 | 2025-09-09 | Yes. Card: supports only instruct (non-thinking) mode. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct |
| Meta | `meta-llama/Llama-3.2-1B-Instruct` | 1.24 dense | 2 | 2024-09-18 | Yes by model type. Gated; card unread. | Llama 3.2 Community License | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct |
| Meta | `meta-llama/Llama-3.2-3B-Instruct` | 3.21 dense | 6 | 2024-09-18 | Yes by model type. Gated; card unread. | Llama 3.2 Community License | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct |
| Meta | `meta-llama/Llama-3.1-8B-Instruct` | 8.03 dense | 16 | 2024-07-18 | Yes by model type. Gated; card unread. | Llama 3.1 Community License | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Meta | `meta-llama/Llama-3.3-70B-Instruct` | 70.6 dense | 141 | 2024-11-26 | Yes by model type. Gated; card unread. | Llama 3.3 Community License | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| Meta | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 108.6 / 17 | 217 | 2025-04-02 | Yes by model type. Gated; card unread. | Llama 4 Community License | https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct |
| Google | `google/gemma-3-1b-it` | 1.00 dense | 2 | 2025-03-10 | Yes by model type. Gated; card unread. | Gemma license | https://huggingface.co/google/gemma-3-1b-it |
| Google | `google/gemma-3-4b-it` | 4.30 dense | 9 | 2025-02-20 | Yes by model type. Gated; card unread. | Gemma license | https://huggingface.co/google/gemma-3-4b-it |
| Google | `google/gemma-3-12b-it` | 12.2 dense | 24 | 2025-03-01 | Yes by model type. Gated; card unread. | Gemma license | https://huggingface.co/google/gemma-3-12b-it |
| Google | `google/gemma-3-27b-it` | 27.4 dense | 55 | 2025-03-01 | Yes by model type. Gated; card unread. | Gemma license | https://huggingface.co/google/gemma-3-27b-it |
| Mistral AI | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24.0 dense | 48 | 2025-06-19 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506 |
| Mistral AI | `mistralai/Ministral-3-3B-Instruct-2512` | 3.85 dense | 5 | 2025-10-31 | Yes. Separate `-Reasoning-2512` weights exist. | Apache-2.0 | https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512 |
| Mistral AI | `mistralai/Ministral-3-8B-Instruct-2512` | 8.92 dense | 10 | 2025-10-31 | Yes. Separate `-Reasoning-2512` weights exist. | Apache-2.0 | https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 |
| Mistral AI | `mistralai/Ministral-3-14B-Instruct-2512` | 13.9 dense | 16 | 2025-10-31 | Yes. Separate `-Reasoning-2512` weights exist. | Apache-2.0 | https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512 |
| Microsoft | `microsoft/phi-4` | 14.7 dense | 29 | 2024-12-11 | Yes. The card has no reasoning mode. | MIT | https://huggingface.co/microsoft/phi-4 |
| Microsoft | `microsoft/Phi-4-mini-instruct` | 3.84 dense | 8 | 2025-02-19 | Yes. The card has no reasoning mode. | MIT | https://huggingface.co/microsoft/Phi-4-mini-instruct |
| Ai2 | `allenai/Olmo-3-7B-Instruct` | 7.30 dense | 15 | 2025-11-19 | Yes. Separate `Olmo-3-7B-Think` weights exist. | Apache-2.0 | https://huggingface.co/allenai/Olmo-3-7B-Instruct |
| Ai2 | `allenai/Olmo-3.1-32B-Instruct` | 32.2 dense | 64 | 2025-12-10 | Yes. Separate `Olmo-3.1-32B-Think` weights exist. | Apache-2.0 | https://huggingface.co/allenai/Olmo-3.1-32B-Instruct |
| IBM | `ibm-granite/granite-4.0-micro` | 3.40 dense | 7 | 2025-09-16 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.0-micro |
| IBM | `ibm-granite/granite-4.0-h-tiny` | 6.94 / ~1 (name) | 14 | 2025-09-16 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.0-h-tiny |
| IBM | `ibm-granite/granite-4.0-h-small` | 32.2 / ~9 (name) | 64 | 2025-09-16 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.0-h-small |
| IBM | `ibm-granite/granite-4.1-3b` | 3.40 dense | 7 | 2026-04-06 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.1-3b |
| IBM | `ibm-granite/granite-4.1-8b` | 8.79 dense | 18 | 2026-04-06 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.1-8b |
| IBM | `ibm-granite/granite-4.1-30b` | 28.9 dense | 58 | 2026-04-06 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ibm-granite/granite-4.1-30b |
| TII | `tiiuae/Falcon-H1-0.5B-Instruct` | 0.52 dense | 1 | 2025-05-01 | Yes. The card has no reasoning mode. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct |
| TII | `tiiuae/Falcon-H1-1.5B-Instruct` | 1.55 dense | 3 | 2025-05-01 | Yes. The card has no reasoning mode. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1-1.5B-Instruct |
| TII | `tiiuae/Falcon-H1-3B-Instruct` | 3.15 dense | 6 | 2025-05-01 | Yes. The card has no reasoning mode. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct |
| TII | `tiiuae/Falcon-H1-7B-Instruct` | 7.59 dense | 15 | 2025-05-01 | Yes. The card has no reasoning mode. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct |
| TII | `tiiuae/Falcon-H1-34B-Instruct` | 33.6 dense | 67 | 2025-05-01 | Yes. The card has no reasoning mode. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct |
| Hugging Face | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71 dense | 3 | 2024-10-31 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct |
| Baidu | `baidu/ERNIE-4.5-0.3B-PT` | 0.36 dense | 1 | 2025-06-28 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT |
| Baidu | `baidu/ERNIE-4.5-21B-A3B-PT` | 21.9 / ~3 (name) | 44 | 2025-06-28 | Yes. Separate `ERNIE-4.5-21B-A3B-Thinking` weights exist. | Apache-2.0 | https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT |
| Moonshot | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 49.1 / ~3 (name) | 98 | 2025-10-30 | Yes. The card has no reasoning mode. | MIT | https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct |
| Z.ai | `zai-org/GLM-4-9B-0414` | 9.40 dense | 19 | 2025-04-07 | Yes. The card lists separate GLM-Z1 reasoning models. | MIT | https://huggingface.co/zai-org/GLM-4-9B-0414 |
| Z.ai | `zai-org/GLM-4-32B-0414` | 32.6 dense | 65 | 2025-04-07 | Yes. The card lists separate GLM-Z1 reasoning models. | MIT | https://huggingface.co/zai-org/GLM-4-32B-0414 |
| AI21 | `ai21labs/AI21-Jamba2-3B` | 3.03 dense | 6 | 2026-01-06 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/ai21labs/AI21-Jamba2-3B |
| AI21 | `ai21labs/AI21-Jamba2-Mini` | 51.6 / 12 | 103 | 2026-01-06 | Yes. The card describes a non-reasoning model. | Apache-2.0 | https://huggingface.co/ai21labs/AI21-Jamba2-Mini |
| Swiss AI | `swiss-ai/Apertus-8B-Instruct-2509` | 8.05 dense | 16 | 2025-08-13 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509 |
| Swiss AI | `swiss-ai/Apertus-70B-Instruct-2509` | 70.6 dense | 141 | 2025-09-01 | Yes. The card has no reasoning mode. | Apache-2.0 | https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509 |

## Table 3. Reasoning-native checkpoints

| Vendor | Checkpoint | Active (B) | Documented behavior | Documented route to disable reasoning | License | Source |
|---|---|---|---|---|---|---|
| Alibaba | `Qwen/Qwen3-4B-Thinking-2507` | dense | Card: supports only thinking mode; the template adds `<think>` to the generation prompt. | Cannot be disabled by documented means. Separate Instruct-2507 weights exist. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 |
| Alibaba | `Qwen/Qwen3-30B-A3B-Thinking-2507` | 3.3 | Same as above. | Same as above. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507 |
| Alibaba | `Qwen/Qwen3-Next-80B-A3B-Thinking` | ~3 (name) | Card: supports only thinking mode. | Cannot be disabled by documented means. Separate Instruct weights exist. | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking |
| Alibaba | `Qwen/QwQ-32B` | dense | Reasoning-only model. | No switch in the card. | Apache-2.0 | https://huggingface.co/Qwen/QwQ-32B |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, -7B, -14B, -32B, DeepSeek-R1-Distill-Llama-8B, -70B` | dense | Always emits reasoning. The card notes the models sometimes output an empty think block and recommends forcing `<think>` followed by a newline. | No documented off switch; the card treats skipped thinking as a failure mode. | MIT | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |
| OpenAI | `openai/gpt-oss-20b` | 3.6 | Reasoning level `low`, `medium` or `high` set in the system prompt (`Reasoning: high`). | No `none` level documented. | Apache-2.0 | https://huggingface.co/openai/gpt-oss-20b |
| OpenAI | `openai/gpt-oss-120b` | 5.1 | Same as above. Card: fits one 80 GB GPU. | No `none` level documented. | Apache-2.0 | https://huggingface.co/openai/gpt-oss-120b |
| Microsoft | `microsoft/Phi-4-reasoning, Phi-4-reasoning-plus (14.7B), Phi-4-mini-reasoning (3.8B)` | dense | Card: always use the ChatML template with the provided reasoning system prompt. | No documented off switch. Non-reasoning `phi-4` and `Phi-4-mini-instruct` are separate weights. | MIT | https://huggingface.co/microsoft/Phi-4-reasoning |
| Mistral AI | `mistralai/Magistral-Small-2509 (24B)` | dense | The reasoning prompt is supplied in the system prompt. | No documented off switch. Mistral-Small-3.2 is a separate non-reasoning checkpoint. | Apache-2.0 | https://huggingface.co/mistralai/Magistral-Small-2509 |
| Mistral AI | `mistralai/Ministral-3-3B-Reasoning-2512, -8B-, -14B-` | dense | Reasoning variants use a provided system prompt file. | No documented off switch. Instruct variants are separate weights. | Apache-2.0 | https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512 |
| Ai2 | `allenai/Olmo-3-7B-Think, Olmo-3.1-32B-Think` | dense | Thinking models with a default system prompt. | No documented off switch. Instruct variants are separate weights. | Apache-2.0 | https://huggingface.co/allenai/Olmo-3-7B-Think |
| NVIDIA | `nvidia/Nemotron-Cascade-8B-Thinking, Nemotron-Cascade-14B-Thinking` | dense | Card code comment: only thinking mode is supported (`enable_thinking=True`). | Cannot be disabled by documented means. | NVIDIA Open Model License | https://huggingface.co/nvidia/Nemotron-Cascade-8B-Thinking |
| TII | `tiiuae/Falcon-H1R-7B` | dense | Generates a `<think>` block before the answer. | No switch in the card. | Falcon-LLM license | https://huggingface.co/tiiuae/Falcon-H1R-7B |
| Baidu | `baidu/ERNIE-4.5-21B-A3B-Thinking` | ~3 (name) | Thinking variant of ERNIE-4.5-21B-A3B. | No switch in the card; the non-thinking weights are `ERNIE-4.5-21B-A3B-PT`. | Apache-2.0 | https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking |
| Kakao | `kakaocorp/kanana-2-30b-a3b-thinking-2601` | ~3 (name) | Thinking checkpoint released beside an Instruct checkpoint. | No switch found in the card. | kanana-license (text not read) | https://huggingface.co/kakaocorp/kanana-2-30b-a3b-thinking-2601 |
| AI21 | `ai21labs/AI21-Jamba-Reasoning-3B` | dense | Reasoning-only model. | No switch in the card. | Apache-2.0 | https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B |
| Xiaomi | `XiaomiMiMo/MiMo-7B-RL-0530` | dense | Reasoning model. The chat template has an `enable_thinking` branch that prefills an empty think block when false. | Template-only flag; the card does not document it. | MIT | https://huggingface.co/XiaomiMiMo/MiMo-7B-RL-0530 |
| Z.ai | `zai-org/GLM-Z1-9B-0414, GLM-Z1-32B-0414` | dense | Deep-thinking reasoning models. | No switch in the card; the GLM-4-0414 chat models are separate weights. | MIT | https://huggingface.co/zai-org/GLM-Z1-9B-0414 |
| StepFun | `stepfun-ai/Step-3.5-Flash (199B MoE, outside the size envelope)` | 11 | The template ends the generation prompt with `<think>`. | No switch in the card or template. | Apache-2.0 | https://huggingface.co/stepfun-ai/Step-3.5-Flash |
| MiniMax | `MiniMaxAI/MiniMax-M2 (229B MoE, outside the size envelope)` | 10 | The template ends the generation prompt with `<think>`; interleaved thinking. | No switch in the card. | Modified MIT | https://huggingface.co/MiniMaxAI/MiniMax-M2 |
| Z.ai (after window) | `zai-org/GLM-5.3, GLM-5.3-Flash (2026-08-25)` | unverified | docs.z.ai: forced thinking. | docs.z.ai: cannot be disabled. | unverified | https://huggingface.co/zai-org/GLM-5.3 |
| Liquid AI (after window) | `LiquidAI/LFM2.5-2.6B (2026-07-28)` | dense | Card: pure reasoning model that always thinks; the template adds `<think>`. | Cannot be disabled by documented means. | LFM Open License (HF tag: other; text not read) | https://huggingface.co/LiquidAI/LFM2.5-2.6B |

No reasoning-native checkpoint in Table 3 has a documented route to disable reasoning. The cards do not describe prefilling an empty think block for these checkpoints; the DeepSeek-R1 card reports that those models skip thinking on some queries and treats this as degraded behavior. Several have sibling non-reasoning weights from the same vendor (Qwen3 Instruct-2507, Ministral-3 Instruct, Olmo-3 Instruct, ERNIE-4.5-PT, GLM-4-0414, phi-4). gpt-oss exposes three effort levels and no off level.

## Count against the rule

Rule: at least 10 eligible checkpoints from at least three families expose a documented controllable switch. Eligibility criteria beyond the switch (external-index freeze, comparable scoring, pinning) are not assessed here.

| Quantity | Count |
|---|---|
| Checkpoints in Table 1 (inside window and size envelope) | 73 |
| Evidence class C (card describes the switch) | 70 |
| Evidence class O (vendor documentation outside the card) | 0 |
| Evidence class T (template or sample code only) | 3 |
| Distinct vendor families in Table 1 | 17 |
| Distinct vendor families among class C rows | 17 |
| Distinct model lines in Table 1 (Qwen3, Qwen3.5, Qwen3.6, Gemma 4, ...) | 23 |
| Table 1 rows whose off mode is verified in the chat template (code P) | 24 |
| Table 1 rows under Apache-2.0 or MIT | 43 |
| Class C rows under Apache-2.0 or MIT | 40, from 13 vendor families |
| Checkpoints in Table 1b (outside envelope, switch documented) | 11 |

Vendor families in Table 1: Alibaba, ByteDance Seed, Deep Cogito, DeepSeek, Google, Hugging Face, IBM, LG AI Research, Mistral AI, NAVER, NVIDIA, Nous Research, OpenBMB, Sarvam AI, Shanghai AI Lab, Tencent, Z.ai.

Lineage notes: Hermes 4 and Cogito are fine-tunes of Qwen, Llama and Seed-OSS bases; the Llama-Nemotron rows derive from Llama; Nemotron-Cascade derives from Qwen3 and Nemotron-3 bases. Counted by base model, fewer families remain.

## External indices

Facts as read on 2026-10-03. Coverage statements for JavaScript-rendered pages come from page-text extraction and cover the rows visible to that extraction.

| Index | Maintainer | URL | Last update seen | Aggregate content | GSM8K or math in aggregate | Coverage of the open models above |
|---|---|---|---|---|---|---|
| Artificial Analysis Intelligence Index v4.3.2 | Artificial Analysis | https://artificialanalysis.ai/methodology/intelligence-benchmarking | Version 4.3.2 "September 2026 to current" (methodology page) | 10 evaluations: AA-Briefcase v1.1, GDPval-AA v2.1, AutomationBench-AA, Terminal-Bench 4.0, SciCode, AA-LCR v1.1, AA-Omniscience, Humanity's Last Exam, GDP.pdf, CritPt | No GSM8K. AIME 2025 and MATH-500 are listed as legacy evaluations outside the index. HLE and CritPt contain math-related items | Leaderboard page (https://artificialanalysis.ai/leaderboards/models) lists Qwen3.5 (9B, 122B, 397B), Qwen3.6-35B, Qwen3-Next-80B, Gemma 4 (E2B, E4B, 12B, 26B, 31B), Nemotron 3 (Nano, Super), gpt-oss 20B and 120B, Mistral Small 4, Llama 3.x and 4 Scout, EXAONE, Granite, DeepSeek V4 Flash. Not shown on the fetched page: GLM-4.5-Air, GLM-4.7-Flash, SmolLM3, Hunyuan, Seed-OSS. Index values on that page for the open models above fall between roughly 5 and 18, except DeepSeek V4 Flash at 25 |
| Epoch Capabilities Index (ECI) | Epoch AI | https://epoch.ai/benchmarks/eci | Data files marked "Updated Oct. 3, 2026" | Latent-capability fit over 53 listed benchmarks; a model needs at least 4 benchmark results to receive a general ECI | Yes. The benchmark list includes GSM8K (model-developer reported), MATH Level 5, OTIS Mock AIME and FrontierMath among 53 inputs; domain ECIs (math, software engineering, cyber) exist | Scores file (https://epoch.ai/data/eci_scores.csv, 370 rows per the extraction) has rows for Qwen3 (8B, 14B, 30B, 32B, 235B), Qwen3.5 (9B, 35B), Qwen3.6 (27B, 35B), Gemma 3 (4B, 12B, 27B), Gemma 4 (26B, 31B), gpt-oss 20b and 120b, Mistral Small 3 to 3.2, Llama 3.x and 4, DeepSeek V3 and V4, Phi-4, Nemotron 3 Ultra. Absent: GLM-4.5-Air, GLM-4.7-Flash, SmolLM3, Hunyuan, Seed-OSS, EXAONE, Granite |
| Arena text leaderboard | Arena (formerly LMArena) | https://arena.ai/leaderboard/text | "Oct 2, 2026" shown on the page | Human pairwise preference votes | No GSM8K or math benchmark; the page has a Math category filter | Page lists Qwen3.5-27B, Qwen3.5-122B-A10B, Gemma-4-31b, Gemma-4-26b-a4b, GLM-5 and GLM-4.5, DeepSeek V4 variants, Mistral-large-3, Nemotron-3-Ultra-550B. Smaller checkpoints of Table 1 were not visible in the extraction |
| HELM Capabilities | Stanford CRFM | https://crfm.stanford.edu/helm/capabilities/latest/ (scenarios from https://crfm.stanford.edu/2025/03/20/helm-capabilities.html) | Blog dated 2025-03-20 (v1.0.0); the current leaderboard page did not render, so its present version is unverified | Mean over 5 scenarios: MMLU-Pro, GPQA, IFEval, WildBench, Omni-MATH | No GSM8K. Omni-MATH is a math scenario | The v1.0.0 post evaluates Llama 3.1 Instruct Turbo, Mistral Instruct v0.3, Mixtral, Qwen2 Instruct Turbo and DeepSeek v3. Coverage of 2025-2026 checkpoints is unverified |
| Open LLM Leaderboard v2 | Hugging Face | https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard | Space last modified 2026-05-27 (status RUNNING); the `contents` results dataset was last modified 2025-03-20; no retirement notice was found on the primary pages read | IFEval, BBH, MATH Level 5, GPQA, MuSR, MMLU-Pro | No GSM8K. MATH Level 5 is included. The archived v1 (June 2024) included GSM8K 5-shot | Row queries on the `contents` dataset returned rows for Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct, gemma-2-27b-it, phi-4 and DeepSeek-R1-Distill-Qwen-7B, and none for Qwen3-8B, SmolLM3-3B and Mistral-Small-24B-Instruct-2501. Queries for Llama-3.1-8B-Instruct, gemma-3-27b-it, Qwen3-32B and gpt-oss-20b failed with server errors and are unverified |
| LiveBench | LiveBench team | https://livebench.ai | Repository changelog newest entry 2026-01-08; repository last push 2026-09-29 | Monthly-refreshed questions; categories reasoning, coding, agentic coding, math, data analysis, language, instruction following (category list from a third-party summary and the repository) | No GSM8K found in the repository README or changelog; has math tasks (competition and olympiad questions, integrals) | The leaderboard page is JavaScript-rendered and was not readable; open-model coverage is unverified |
| OpenCompass CompassRank | Shanghai AI Lab / OpenCompass | https://rank.opencompass.org.cn/leaderboard-llm | Page content not readable (JavaScript-rendered) | Aggregate composition not read | The OpenCompass framework lists GSM8K among its supported datasets; whether the leaderboard aggregate uses it is unverified | Unverified |

## Unverified items

- Gated model cards not read: Llama 3.x and 4 Scout, Gemma 3, `CohereLabs/command-a-reasoning-08-2025` (switch confirmed from docs.cohere.com only), `swiss-ai/Apertus-v1.5-*`.
- License texts read: Tencent Hunyuan (A13B and 7B repositories), EXAONE (4.0-32B and 4.5-33B repositories), HyperCLOVA X SEED 14B Think (first lines only). All other license names come from Hugging Face metadata or card headers.
- Off-mode code P was checked on the templates of these checkpoints: Qwen3-8B, Qwen3.5-9B, GLM-4.5-Air, GLM-4.7-Flash, Nemotron-Nano-9B-v2, MiniCPM4.1-8B, Gemma-4-31B-it. Rows marked P for sibling checkpoints rely on the sibling template having the same structure (template lengths match for the Qwen3.5 and Gemma 4 siblings).
- Qwen3.5-0.8B and 2B default to non-thinking; Qwen3.5-4B and larger default to thinking (card and template read for each).
- Active-parameter values marked "(name)" were not found in card text.
- Not examined: vision-language releases with a switch (for example ERNIE-4.5-VL and GLM-4.6V-Flash), Kimi, MiMo-V2, Ling and Ring, LongCat, Apriel and community fine-tunes. The tables cover the vendors named in them and are not exhaustive.
- Index coverage for Artificial Analysis, ECI and Arena was extracted from rendered pages by a text-extraction tool and may omit rows. ECI coverage was read from the CSV through a page fetch; the file was not saved.
- The Open LLM Leaderboard row-query results come from the Hugging Face dataset-viewer filter endpoint; several queries returned server errors.

## Sources

- Model cards and chat templates: https://huggingface.co/ followed by the checkpoint id, for every checkpoint in the tables (column Source).
- Z.ai thinking mode documentation: https://docs.z.ai/guides/capabilities/thinking-mode
- Cohere reasoning documentation: https://docs.cohere.com/docs/reasoning
- DeepSeek-V4 encoding script (thinking_mode values): https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py
- Tencent Hunyuan license text: https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/LICENSE and https://github.com/Tencent-Hunyuan/Hunyuan-7B/blob/main/LICENSE.txt
- EXAONE license text: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/blob/main/LICENSE
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- Epoch Capabilities Index: https://epoch.ai/benchmarks/eci and https://epoch.ai/data/eci_scores.csv
- Arena text leaderboard: https://arena.ai/leaderboard/text
- HELM Capabilities: https://crfm.stanford.edu/2025/03/20/helm-capabilities.html
- Open LLM Leaderboard: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard, https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/about and https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive
- LiveBench: https://github.com/LiveBench/LiveBench
- OpenCompass: https://rank.opencompass.org.cn/leaderboard-llm
