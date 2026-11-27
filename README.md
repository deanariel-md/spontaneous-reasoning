# Spontaneous reasoning reorders language models' gains from chain-of-thought: code

This repository holds the code of the paper "Spontaneous reasoning reorders language models’ gains from chain-of-thought". It builds the prompts, runs the models, scores the outputs and computes every number and table of the paper and its Supplementary Information. It also holds `pilot_report`, a small tool that reports the share of long requested answers, the reliability of gains across templates, and who benefits under each baseline, from a pilot of your own.

The study measures the gain of chain-of-thought prompting against two direct-answer baselines. In the requested baseline the model is asked for the final answer only. In the forced baseline the opening of the answer is written into the reply and the model may add at most 16 tokens (64 on HotpotQA). The benchmarks are GSM8K, MATH-500, BIG-Bench Hard and HotpotQA.

The model outputs, scores and per-draw accuracies are not part of this release.

## Layout

| Folder | Contents |
|---|---|
| `src/` | library: estimators, prompt banks and samplers, request building, generation, scoring, analyses |
| `scripts/` | command-line steps: request builders, generation runners, API runs, scorers, analyses |
| `config/` | model registries, decoding settings, screening rules, benchmark and bank settings |
| `manifests/` | item lists, template banks, samplers (the drawn templates), panels, and the prediction rules saved before the validation draws were scored |
| `paper/` | `make_numbers.py` and its macro modules, which write `paper/numbers.tex` |
| `pilot_report/` | the pilot tool, its README, examples and example builder |
| `tests/` | unit tests |

Every script runs from the repository root and finds `data/` and `results/` at the root.

## Install

Python 3.12 or later. The analyses ran on Python 3.14.

```
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the packages that the analyses and tests import. Some steps need more packages; they are listed at the end of `requirements.txt`. Local generation needs `vllm==0.30.0` and `torch==2.13.0` on a CUDA GPU.

The prompt-bank builder and the two paraphrase steps (`scripts/build_bank_candidates.py`, `scripts/paraphrase_d2.py`, `scripts/select_answer_format_d2.py`) run PromptSuite in an environment of its own, `.venv-promptsuite` at the repository root, because its declared requirements would downgrade packages pinned here. Make it with Python 3.12 and install without dependencies:

```
python3.12 -m venv .venv-promptsuite
.venv-promptsuite/bin/python -m pip install --no-deps -r requirements-promptsuite.txt     # Windows: .venv-promptsuite\Scripts\python.exe
```

## The study's outputs

The analyses read the study's model outputs, scores and per-draw accuracies. They are available from the corresponding author on request, as a data package that stores one folder per collection: the outputs of all models in one table, the prompts once (they are the same for every model) and the score tables. Its folder `analysis_inputs/` holds the other files the analyses read, such as run logs, the API runs' reply records and choices saved before the validation draws were scored. Its README describes every file.

The analyses read the layout of the study's own runs under `data/` and `results/`. `scripts/restore_layout.py` writes that layout from the unpacked package. Unpack the package into a folder of its own, for example `../spontaneous_reasoning_data`, and run from the repository root:

```
python scripts/restore_layout.py ../spontaneous_reasoning_data --only inputs
python -I scripts/fetch_bbh.py
python -I scripts/fetch_math.py
python -I scripts/fetch_hotpot.py
python -c "from src import gsm8k; gsm8k.download()"
python scripts/restore_layout.py ../spontaneous_reasoning_data
```

The first command copies the analysis inputs. `fetch_hotpot.py` checks the HotpotQA token lengths among them. The fetch scripts download the public benchmark files at the pinned revisions into `data/raw/` and write the item tables. The last command needs them, because the labels of the scored tables carry the hash of the item tables. With the first command it writes about 1,350 files (about 7 GB) in about five minutes and lists each file with its source in `data/restore_record.json`. Where the code needs something the package leaves out, it is rebuilt:

- the request keys (`<template>|<arm>|<item>`, with the suffix some designs add);
- one generation folder per model and run, with one shard and the run's logs;
- score tables split per model where the scorer wrote one file per model;
- the metadata the analyses check before they read a scored table;
- the API runs' raw reply files, from the package's outputs and the reply records in `analysis_inputs/`.

Two more downloads are needed. The token-level analyses read the panel's tokenizer files: `python scripts/run_instability.py`, run once without `--no-download`, downloads them into `data/tokenizers/` (some models are gated on Hugging Face). The list below passes `--no-download`, because the files are then in place. The check of DeepSeek V4.1-Flash's forced replies renders its prompts with the model's chat template:

```
python -c "import sys; sys.path.insert(0, 'scripts'); import nebius_prefill_test as t; t.fetch_files('deepseek-ai/DeepSeek-V4.1-Flash-1017')"
```

## What some records hold

- `manifests/external/eci_scores_20261030.csv` keeps the open-weight rows of Epoch AI's file. The panel rule reads only those rows.
- The vendors' model cards are not included. For each card the code reads, `manifests/external/vendor_cards/<model>/generation_config.json` keeps the three sampling values it reads. `config/decoding_thinking_vendor.yaml` records each card's sha256.
- The human audit's rater sheets are not shared. Which rater read which answer is in `manifests/human_audit/assignment.csv`.
- The instability, decision, reliability-map and noise analyses also run a DOVE setting: five-shot against zero-shot on the MMLU block of DOVE_Lite. Its tables are built from DOVE_Lite (Hugging Face `nlphuji/DOVE_Lite`, CDLA-Permissive-2.0). The manuscript reports none of its values.

## Reproduce the paper from the outputs

After the restore step, the commands below rebuild every number and table of the paper. They were run in this order on a fresh copy of this repository and the data package, and the times are from that run on one desktop computer. In that run the public files above were copied from the study's own copies, not downloaded. Each step reads only the package, those files and the outputs of earlier steps.

```
# DOVE setting and the reproduction tables (about 1 min)
python scripts/run_dataset1.py
python scripts/run_dataset1_tasks.py
python scripts/run_reproduction.py --scored data/processed/reproduction/numerical_scored.parquet data/processed/reproduction/transport_scored.parquet
python scripts/run_cheng_panel.py

# Main panel (about 35 min)
python scripts/run_dataset2.py --scored data/processed/gsm8k/scored.parquet
python scripts/run_dataset2.py --panel extension
python scripts/run_dataset2.py --task bbh --scored data/processed/bbh/scored.parquet
python scripts/run_dataset2.py --task math --scored data/processed/math/scored.parquet
python scripts/run_dataset2_tasks.py
python scripts/run_nonoverlap_validation.py
python scripts/run_nonoverlap_validation.py --task bbh
python scripts/run_reliability_map.py --bbh --gsm8k30 --math
python scripts/run_noise_anatomy.py --settings dove_mmlu gsm8k gsm8k30 bbh math
python scripts/run_instability.py --no-download
python scripts/run_instability_tasks.py
python scripts/run_transfer_tasks.py
python scripts/run_index_checks.py
python scripts/run_forced_direct.py
python scripts/run_forced_direct.py --tasks math
python scripts/run_forced_constrained.py
python scripts/run_content.py
python scripts/run_outcomes.py
python scripts/run_outcomes.py --scoring lenient
python scripts/run_outcomes.py --scoring content
python scripts/run_outcomes.py --tasks math
python scripts/run_outcomes.py --scoring lenient --tasks math
python scripts/run_outcomes.py --scoring content --tasks math
python scripts/run_exemplar.py
python scripts/run_exemplar.py --noise-split
python scripts/run_fewshot.py
python scripts/run_fewshot.py --tasks math
python scripts/analyze_thinking_pilot.py
python scripts/run_thinking.py
python scripts/run_map_panels.py --forced --exemplar --thinking
python scripts/run_pilot_checks.py
python scripts/run_pilots_v3.py
python scripts/run_planning_rule.py
python scripts/run_rule_budgets.py

# Later analyses on the main panel (about 20 min)
python scripts/run_math_prediction.py
python scripts/run_late_generation.py
python scripts/run_heldout_capability.py
python scripts/run_lowr_map.py
python scripts/run_hotpot.py
python scripts/run_reliability_map.py --bbh --gsm8k30 --math --added
python scripts/run_map_panels.py --forced --exemplar --thinking --added
python scripts/run_noise_anatomy.py --settings fewshot_gsm8k fewshot_bbh fewshot_math constrained_bbh hotpot_cot_req hotpot_cot_forced hotpot_fewshot
python scripts/run_hotpot_mech_fresh.py
python scripts/run_arm_wording.py
python scripts/forced_stop_audit.py
python scripts/forced_stop_full_replies.py
python scripts/run_content.py
python scripts/run_specification.py
python scripts/run_patterns.py
python scripts/run_math_primary.py
python scripts/run_baseline_mixture.py
python scripts/run_decision.py
python scripts/run_decision_strict.py
python scripts/run_decision_directions.py
python scripts/run_arm_effect.py
python scripts/run_generality.py
python scripts/run_protocol_ceiling.py
python scripts/run_content_arm_ci.py
python scripts/run_tipping_points.py
python scripts/run_transfer_regret.py
python scripts/run_split_pilot.py
python scripts/run_output_checks.py
python scripts/run_hotpot_content.py
python scripts/run_model_resampling.py
python scripts/run_gsm8k_difficulty.py
python scripts/run_fewshot_requested.py
python scripts/run_readable_sweep.py

# Capability, thresholds, Qwen4-Max, temperature check, audits (about 13 min)
python scripts/run_within_difficulty.py
python scripts/run_cap_shared_axis.py
python scripts/run_cap_reversal.py
python scripts/run_cheng_scales.py
python scripts/threshold_sweep.py
python scripts/threshold_decision.py
python scripts/closed_w.py report
python scripts/closed_w_full.py report
python scripts/closed_w_intervals.py
python scripts/temp_check.py report
python scripts/reading_audit.py analyse
python scripts/run_human_audit.py
python scripts/run_human_audit.py --three-raters

# The larger models (about 7 min)
python scripts/ladder_bbh_forms.py
python scripts/check_mistral_bbh_forced.py
python scripts/check_nebius_forced.py > results/frontier_panel/nebius_forced_check.json
python scripts/nebius_panel_report.py
python scripts/ladder_analysis.py
python scripts/arm_wording_within_difficulty.py
python scripts/ladder_orderings.py
python scripts/statement_values.py
python scripts/release_dates.py run

# Baseline parts, five-shot w, Fig. 2c on validation templates
python scripts/run_attr_flips.py
python scripts/run_fewshot_w.py --content
python scripts/run_fewshot_w.py
python scripts/run_bmix_val.py

# Numbers and tables (about 8 min)
python scripts/write_curve_tables.py
python paper/make_numbers.py
python scripts/threshold_sweep.py
python scripts/run_generality.py
python scripts/ladder_arms_table.py
python scripts/verify_ladder_analysis.py
```

`paper/make_numbers.py` writes `paper/numbers.tex`, one LaTeX macro per number. `scripts/run_generality.py` and `scripts/threshold_sweep.py` compare some of their values with the macros of `paper/numbers.tex` when it exists; before the first `make_numbers.py` run they skip that comparison, so both run again after it. `python scripts/compare_numbers_tex.py BEFORE AFTER` compares two `numbers.tex` files name by name.

Rules fitted on the discovery draws are saved in `manifests/predictions_*.json` before the validation draws are scored. A script that would rewrite a saved rule refuses unless it is told to (for example `--overwrite-frozen`).

| Item | Made by | Output |
|---|---|---|
| Every number in the main text and the SI | `paper/make_numbers.py` | `paper/numbers.tex` |
| SI table: the cutoff for a long answer | `scripts/threshold_sweep.py` | `paper/tables/threshold_sweep.tex` |
| SI table: reading the requested answers | `scripts/reading_audit.py analyse` | `paper/tables/reading_audit.tex` |
| SI table: accuracy of each larger model in each condition | `scripts/ladder_arms_table.py` | `paper/tables/ladder_arms.tex` |
| SI tables: thinking-mode pilot; native reasoning per checkpoint | `paper/make_numbers.py` | `paper/tables/thinking_pilot.tex`, `paper/tables/thinking_runs.tex` |
| Every other SI table | its values are macros of `paper/numbers.tex` (see below) | |

Each macro of `numbers.tex` comes from one analysis output. The prefix of a macro name tells which:

| Prefix | Analysis (script) |
|---|---|
| `gt`, `bh`, `mt`, `dt` | panel analyses per benchmark (`scripts/run_dataset2.py`, `scripts/run_dataset2_tasks.py`) |
| `ddir`, `dec` | decision costs under each baseline (`scripts/run_decision_directions.py`, `scripts/run_decision.py`) |
| `bmix` | the requested baseline as a mixture of short and long answers (`scripts/run_baseline_mixture.py`) |
| `pceil` | protocol ceiling, agreement limits of the two baselines (`scripts/run_protocol_ceiling.py`, `scripts/run_content_arm_ci.py`) |
| `aw` | rewording of the request and the trigger (`scripts/run_arm_wording.py`) |
| `hq`, `fmt` | HotpotQA (`scripts/run_hotpot.py`, `scripts/run_hotpot_content.py`, `scripts/hotpot_fair.py`) |
| `cs` | content scoring classes (`scripts/run_content.py`) |
| `crv`, `cap` | capability slopes under each baseline (`scripts/run_cap_reversal.py`, `scripts/run_cap_shared_axis.py`) |
| `thr` | threshold sweep and threshold decision (`scripts/threshold_sweep.py`, `scripts/threshold_decision.py`) |
| `lad`, `comb` | the six larger models (`scripts/ladder_analysis.py`, `scripts/ladder_orderings.py`) |
| `cw` | Qwen4-Max (`scripts/closed_w.py`, `scripts/closed_w_full.py`, `scripts/closed_w_intervals.py`) |
| `lab` | reading audit (`scripts/reading_audit.py`) |
| `hra`, `hrt` | human audit with two and three raters (`scripts/run_human_audit.py`) |
| `temp` | temperature 0.7 check (`scripts/temp_check.py`) |
| `fsr`, `fs` | five-shot requested answers; forced-stop completion check (`scripts/run_fewshot_requested.py`, `scripts/forced_stop_full_replies.py`) |
| `thi`, `thp` | native reasoning and its pilot (`scripts/run_thinking.py`, `scripts/analyze_thinking_pilot.py`) |
| `diff`, `reword` | difficulty within a model (`scripts/run_within_difficulty.py`) |
| `attr` | which part of the difference between the baselines makes the reversals: answers given at length or briefly (`scripts/run_attr_flips.py`) |
| `fsW` | w under five-shot answering, and its link to the five-shot effect under each baseline (`scripts/run_fewshot_w.py`) |
| `bmixVal` | Fig. 2c's counts on validation draws whose templates no discovery draw used (`scripts/run_bmix_val.py`) |
| `wck`, `splitp`, `rsw`, `pat`, `spec`, `gen`, `cheng`, `inst`, `eci`, `tr`, `nonov`, `mp` | robustness checks (`scripts/run_output_checks.py`, `scripts/run_split_pilot.py`, `scripts/run_readable_sweep.py`, `scripts/run_patterns.py`, `scripts/run_specification.py`, `scripts/run_generality.py`, `scripts/run_cheng_panel.py`, `scripts/run_instability.py`, `scripts/run_index_checks.py`, `scripts/run_transfer_tasks.py`, `scripts/run_nonoverlap_validation.py`, `scripts/run_math_primary.py`) |
| `bbhConflict` | BIG-Bench Hard boxes whose letter and text name different options (read from `results/conflict_sensitivity/conflict_sensitivity.json`, restored from the package's `analysis_inputs/`) |

## Run the study from the start

The steps below make the study's outputs. They need GPUs and, for three models, paid API access.

1. Items: `scripts/fetch_math.py`, `scripts/fetch_bbh.py`, `scripts/fetch_hotpot.py` (GSM8K is read through `src/gsm8k.py`); `scripts/hotpot_lengths.py` for HotpotQA's length rule.
2. Panel: `scripts/select_panel_d2.py` chooses the panel from the Epoch Capabilities Index; `scripts/draft_registry.py` drafts the registry entries of `config/models.yaml`.
3. Template banks: `scripts/build_bank_candidates.py` (PromptSuite), `scripts/paraphrase_d2.py`, `scripts/screen_d2.py` (screening on training items), `scripts/select_answer_format_d2.py`, `scripts/freeze_bank_d2.py`; draws: `scripts/freeze_sampler_d2.py`; output caps: `scripts/draw_tranche_d2.py`, `scripts/analyze_tranche.py`.
4. Requests: `scripts/build_requests.py` and the other `scripts/build_*.py` (forced, five-shot, exemplars, HotpotQA, native reasoning, rewording, larger models).
5. Generation on GPUs: `scripts/generate_d2.py --requests <file> --model-key <key> --out-dir <dir>` (vLLM, greedy decoding, settings from `config/decoding*.yaml`); `scripts/capcheck_generate.py`, `scripts/thinking_vendor_generate.py` and `scripts/temp_check.py` for their runs.
6. API runs: `scripts/closed_w.py` and `scripts/closed_w_full.py` (Qwen4-Max), `scripts/nebius_panel.py` (DeepSeek V4.1-Flash), `scripts/mistral_ladder.py` (Mistral Large 3), `scripts/reading_audit.py judge` (the reading audit). Each paid phase needs an explicit approval flag and `--limit-usd` (see each script's `--help`), and refuses to spend more.
7. Scoring: `scripts/score_d2.py` and the other `scripts/score_*.py`, then `scripts/score_content.py`, `scripts/score_content_added.py` and `scripts/score_hotpot_content.py` for content scoring.

Each script documents its inputs, outputs and flags in its docstring and `--help`.

### API keys

Each API script reads its key from the provider's standard environment variable and never prints or stores it.

| Script | Environment variable |
|---|---|
| `scripts/closed_w.py`, `scripts/closed_w_full.py` | `OPENROUTER_API_KEY` |
| `scripts/nebius_panel.py`, `scripts/nebius_prefill_test.py` | `NEBIUS_API_KEY` |
| `scripts/mistral_ladder.py` | `MISTRAL_API_KEY` |
| `scripts/reading_audit.py judge` | `ANTHROPIC_API_KEY` |
| `scripts/build_bank_candidates.py` (PromptSuite paraphrases) | the variable PromptSuite reads for the chosen platform, for example `TOGETHER_API_KEY` |

Downloads from Hugging Face use the token of `hf auth login` for gated models.

## The pilot tool

`pilot_report/pilot_report.py` is one file that needs only numpy and pandas. From the repository root:

```
python pilot_report/pilot_report.py pilot_report/examples/gsm8k29_scores.csv --baseline direct_answer --intervention zero_shot_cot
python pilot_report/pilot_report.py pilot_report/examples/gsm8k29_two_baselines_scores.csv --baseline direct_answer --forced forced_direct --intervention zero_shot_cot --tokens pilot_report/examples/gsm8k29_tokens.csv --capability pilot_report/examples/gsm8k29_capability.csv --items pilot_report/examples/gsm8k29_pilot_items.csv --decision
```

`pilot_report/README.md` describes the input files, the output and how each number is computed.

## Tests

```
python -m pytest -q
```

Tests that need the study's outputs skip when they are not restored.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). Third-party data, tools and models are credited in `CREDITS.md`. To cite the work, see `CITATION.cff`.
