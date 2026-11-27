# Pilot report

`pilot_report.py` reads the scores of M models on K independently drawn prompt configurations under two arms, a baseline and an intervention (for example a direct answer and chain of thought), and says how reliably those draws order the models by benefit. It prints, and with `--json` saves:

- R, the reliability of benefit across draws, with the signal and noise behind it;
- C, how often two of the pilot's draws order a pair of models the same way;
- the Gaussian reference C_ref = 1/2 + arcsin(R)/π;
- the number of draws each evaluation should average for 80% and 90% agreement;
- an unresolved flag, with its reason, when R gives no draw count;
- each model's mean scores and mean gain over the draws;
- on request, a check that the baseline arm of a chain-of-thought contrast answers directly.

The same quantities are given for accuracy under each arm, for comparison with benefit.

A chain-of-thought contrast can be measured against two direct-answer baselines. In the requested baseline the model is asked for the final answer only, and whatever it writes is scored. In the forced baseline the opening of the answer is written into the model's reply and the model may add at most 16 tokens. A model's w is the share of its requested replies longer than 16 tokens. Given both baselines (`--forced`), the report adds:

- R and C for benefit and accuracy under the forced baseline;
- per model, the gain under each baseline, the difference between the two gains and, with token counts, w;
- who benefits under each baseline: the capability slope of gain on the model's own-baseline accuracy, and with `--capability` on a capability score such as the Epoch Capabilities Index; the top third of models by mean gain; how many models the two top thirds share; and the share of model pairs that the two baselines order oppositely by gain;
- with item-level scores (`--items`), the split pilot: half of each draw's items under each baseline, the same items for every model, and the top third of models by the resulting gain;
- with `--decision`, under each baseline, the third chosen by own mean gain and the third chosen by the capability slope, how many models they share, and with four or more draws the regret of each choice on draws not used to make it.

`--capability` and `--decision` also work with one baseline. Without `--forced`, `--capability`, `--items` and `--decision`, the report holds only the quantities of the first list.

## Requirements

Python 3 with numpy and pandas. The script is one file and imports nothing else beyond the standard library. It was tested with Python 3.14 (numpy 2.5.3, pandas 3.0.6) and Python 3.12 (numpy 2.4.4, pandas 2.3.3), which print identical reports on the three examples; their JSON files differ in the last digits of a few values (by at most 3e-13).

## Input

Every input is a CSV file. Lines at the top of a file that start with `# ` are comments and are skipped.

Scores: a CSV in long form, one row per model, draw and arm.

| column | meaning |
|---|---|
| `model` | model name |
| `draw` | label of one draw: one prompt configuration applied to every item |
| `arm` | arm name; rows of arms other than those named on the command line are ignored |
| `score` | the model's accuracy on that draw under that arm, in percentage points (0 to 100) |

Other columns are ignored. Every model needs a score on every draw under every arm named. A missing or repeated cell stops the script with an error, and nothing is filled in. The draws should be independent draws from the distribution of configurations a claim is meant to hold over. When a sampler draws with replacement and the same configuration comes up twice, give the two draws different labels: they count as two draws. With two baselines the table holds three arms: the requested baseline (`--baseline`), the forced baseline (`--forced`) and the intervention.

Token counts, for the comparison-arm check and w: a CSV with columns `model`, `arm` and `n_output_tokens`, one row per output. A file with one row per output can be large; one row per (model, arm, length) with a `count` column holding the number of outputs of that length gives the same result. Only the baseline arm's rows are read (with two baselines, the requested baseline's).

Capability scores: a CSV with columns `model` and `capability`, one row per model, for example the Epoch Capabilities Index. A model of the score table without a value is left out of the slope on the capability score, and the report names it.

Item-level scores, for the split pilot: one row per model, draw and item, with columns `model`, `draw`, `item` and either `arm` and `score` (one row per arm) or one column per arm, named as the arms. A score is 1 for a correct answer and 0 for a wrong one; values in between are read as partial credit. The draws and items are the pilot's and need not match the score table's draws, but the models must be the same. The table can take two forms:

- every item scored under both baselines and the intervention: the script assigns half of each draw's items to each baseline (see "How the numbers are computed"); every draw needs the same number of items;
- every item scored under one baseline, the same baseline for every model, and under the intervention: a split pilot already run, whose assignment the table itself gives.

An optional `stratum` column balances the assignment within strata (for example the tasks of BIG-Bench Hard); each stratum of a draw must then hold the same number of items.

## Running it

```
python pilot_report.py scores.csv --baseline direct_answer --intervention zero_shot_cot
python pilot_report.py scores.csv --baseline direct_answer --intervention zero_shot_cot --tokens tokens.csv --token-budget 16 --max-over-budget 0.05 --json report.json
python pilot_report.py scores.csv --baseline direct_answer --forced forced_direct --intervention zero_shot_cot --tokens tokens.csv --capability capability.csv --items items.csv --decision --json report.json
```

| option | default | meaning |
|---|---|---|
| `--baseline` | required | arm name of the comparison condition; with `--forced`, the requested baseline |
| `--intervention` | required | arm name of the prompting strategy |
| `--tokens` | none | token-count CSV; runs the comparison-arm check, and with `--forced` gives w |
| `--token-budget` | 16 | output tokens a direct answer may take; w counts the requested replies longer than this |
| `--max-over-budget` | 0.05 | largest share of baseline outputs above the budget that passes |
| `--forced` | none | arm name of the forced baseline; adds the two-baseline report and the block "who benefits" |
| `--capability` | none | capability CSV; adds the slope of gain on the capability score |
| `--items` | none | item-level CSV; adds the split pilot (needs `--forced`) |
| `--seed` | 20261121 | seed of the split pilot's assignment, and of the random splits of the split-half regret when the draws are too many to use every split |
| `--decision` | off | adds the decision block |
| `--json` | none | write the full report to this file |

On invalid input the script prints the problem and exits with code 2. From Python:

```python
from pilot_report import pilot_report, format_report
res = pilot_report("scores.csv", "direct_answer", "zero_shot_cot", tokens="tokens.csv")
print(format_report(res))
res = pilot_report("scores.csv", "direct_answer", "zero_shot_cot", tokens="tokens.csv", forced="forced_direct",
                   capability="capability.csv", items="items.csv", decision=True)
```

`pilot_report` also accepts pandas DataFrames, and `report_arrays(B, P)` takes the two arms as arrays of draws by models (`report_arrays(B, P, F=F, forced="forced_direct", decision=True)` adds the forced baseline's array F).

## Reading the output

Benefit is the intervention's score minus the baseline's, per model and draw, in percentage points.

- **R** is the share of the variance of benefit across models within one draw that repeats from draw to draw. Near 1, one draw orders the models by benefit much as another draw would. Near 0, the ordering mostly reflects the particular prompt. R is reported as estimated: with few draws or little true variation it can be negative, and it is never set to zero.
- **signal S** is the variance of benefit across models that persists across draws, in pp². **Within-draw variance T** is the variance of one draw's benefits across models, averaged over draws. **T − S** is the part that comes from the prompt draw. R = S / T.
- **C observed** is the share of model-pair comparisons on which two different pilot draws agree, averaged over all pairs of distinct draws and with equal weight on every model pair. A comparison in which either draw ties the pair counts one half, and a difference of at most 1e-9 pp is a tie. With few draws and few models, C moves in coarse steps.
- **C_ref** is the agreement the Gaussian reference expects from R. A gap between C and C_ref means the panel's differences are not well summarized by one R, as happens with few models or with a few large differences among many small ones.
- **R of the K-draw mean** and **C_ref for two K-draw means** apply the reference to the average of the pilot's own K draws, with R_K = K R / (1 + (K − 1) R).
- **Draws for 80% (90%) agreement** is the number of draws each of two evaluations should average so that, under the reference, the two order a model pair the same way 80% (90%) of the time. Taken from a pilot, it is a planning figure with uncertainty of its own.
- **Status** is "unresolved" when R gives no draw count, and the reason is printed below the table. There are three: fewer than two draws; zero within-draw variance, where every model has the same benefit in each draw and R is undefined; and a signal of zero or below, where the draws detect no model differences beyond prompt noise. A quantity with no prompt noise at all is resolved, with a note that one draw suffices.
- A **warning** appears when K < 4. R from two or three draws is imprecise and often negative, so plan from a pilot of at least four.

The per-model table gives each model's mean score under each arm, its mean gain and the standard deviation of its gain across draws. These belong next to any claim about which models gain more.

**Comparison-arm check.** A chain-of-thought contrast measures benefit against a direct answer. When a model reasons in the direct arm anyway, its measured benefit compares reasoning with reasoning. Forcing the direct answer (prefilling the answer box in the assistant turn and allowing at most 16 new tokens) costs at most 16 tokens per item. The check counts, for each model, the baseline outputs longer than the budget and flags the model when they are more than 5% of its outputs: "comparison arm does not do what its name says". A share equal to the maximum passes. The median and maximum lengths help to read a flag. A median far above the budget means the model reasons in the direct arm. A median within the budget with a short maximum means the model mostly puts a sentence around its answer. In both cases the requested direct arm is longer than a forced direct answer; forcing it and rerunning the report with the forced arm as the baseline settles the question. The study's admissibility screen used a looser rule, which flags a model when more than half of its direct answers run past 128 tokens (`--token-budget 128 --max-over-budget 0.5`); all 29 models of the GSM8K example pass it.

**Two baselines.** The main table gains columns for benefit over the forced baseline (benefit/forced) and accuracy under it (acc. forced); benefit/req and acc. req are those of the requested baseline. The per-model table gives the mean accuracy under the three arms, the gain over each baseline with its standard deviation across draws, and their difference, gain/req − gain/forced, which equals forced minus requested accuracy. A large negative difference is typical of a model that reasons in its requested replies, and w shows how often it does. With token counts, the last column is w, the share of the model's requested replies longer than the token budget (16 by default), and the line below the table gives its mean over models. w needs only the lengths of the requested replies.

**Who benefits.** Each column applies one baseline.

- **Capability slope on own-baseline accuracy** is the change in gain per point of the baseline's accuracy across models, in pp per pp. A negative slope says that models with lower accuracy gain more, a positive slope that models with higher accuracy gain more. It is computed from cross-draw moments: the accuracy and the gain set against it come from different draws, so noise that a single draw adds to both does not enter. A (the cross-draw moment of gain and accuracy) and V_x (the cross-draw variance of accuracy) are printed; when V_x is zero or below, the slope is undefined and reported as such. Under the two baselines a model is placed on two different axes, so the two slopes can differ in sign.
- **Slope on the capability score** sets the same gains against one axis shared by both baselines, in pp of gain per unit of the score. Models without a score are left out and named.
- **Top third by mean gain** lists round(M/3) models (10 of 29). Models whose mean gains lie within 1e-9 pp of the cut share the remaining places equally and are printed with their share.
- **Shared by the two top thirds** counts the models chosen under both baselines.
- **Model pairs the two baselines order oppositely** counts the pairs of models for which one model has the larger mean gain under one baseline and the smaller under the other. A pair whose gap is at most 1e-9 pp under either baseline is tied and counts as neither. The second line restricts the count to pairs whose gaps exceed 1, 2, 5 and 10 pp under both baselines.

**Split pilot.** A team that does not know which baseline it will deploy can spend its pilot's baseline calls half on each. The block states the assignment, then gives each model's gain on the pilot's items under the requested baseline alone (gain/req) and under the forced baseline alone (gain/forced), when the table holds both, and its split gain: 100 times the mean over the pilot's items of the intervention's score minus the score under the item's assigned baseline. The split pilot gives the intervention to the top third by split gain. The block then counts the models this third shares with the top third of the score table's gains under each baseline and, when the item table holds both baselines, with the third that the same items give under each baseline alone. A split pilot uses as many baseline calls per model as a pilot on one baseline.

**Decision.** For each baseline, the intervention goes to the top third chosen by one of two rules: the models' own mean gains, or the capability slope, which predicts each model's gain from its mean accuracy along the slope. The block gives how many models the two thirds share and lists both. With four or more draws it gives the split-half regret of each rule: the draws are split into two halves, a third is chosen from one half, and on the other half the mean gain of the best third minus the mean gain of the chosen third, per chosen model, is the regret in pp. Every split of the draws into halves enters, in both directions (six choices from four draws), and the block gives the mean with the smallest and largest values. The best third is found on the same half it is valued on, which makes the regret larger in expectation than the regret against the models' true gains; with two draws per half it is also noisy. A regret can print as −0.00 from floating-point residue, which is reported as it is.

The JSON file holds `M`, `K`, `models`, `draws`, `targets`; one block each for `benefit`, `accuracy_baseline` and `accuracy_intervention` (`R`, `R_status`, `signal`, `within`, `noise`, `C`, `same`, `opposite`, `tie`, `C_ref`, `R_K`, `C_ref_K`, `draws_needed`, `unresolved`, `reason` and, where it applies, `note`); `unresolved` and `unresolved_reason` for benefit; `per_model`; `comparison_arm_check`; and `warnings`. With `--forced` it also holds `forced`, the blocks `benefit_forced` and `accuracy_forced`, the per-model keys `forced_mean`, `gain_forced_mean`, `gain_forced_sd_across_draws`, `gain_difference` and, with token counts, `w`, and the summary `w`. With any of the new options it holds `who_benefits` (per baseline: `slope_on_accuracy` and `slope_on_capability` with `slope`, `A`, `V_x`, `defined` and `reason`, and `top_third` with `weights`, `chosen` and `n_tied`; with two baselines `shared_top_third` and `opposite_orderings`, which names the reversed pairs); with `--items`, `split_pilot` (the per-draw split gains and the assignment of every item, `assignment_cells`); and with `--decision`, `decision`. JSON has no NaN, so an undefined value is written as `null`.

## Example 1: one DOVE benchmark

Five instruction-tuned models on ARC-Challenge (100 items) from the DOVE collection of prompt variations. Zero-shot is the baseline and five-shot the intervention. The four draws are pilot configurations drawn at random from the 2,729 configurations the benchmark shares across models and arms. The benchmark is the first in alphabetical order among the study's DOVE benchmarks; its result played no part in the choice.

```
python pilot_report.py examples/dove_arc_challenge_scores.csv --baseline zero_shot --intervention five_shot
```

```
Pilot report (pilot_report 1.1.0)
  arms: zero_shot (baseline), five_shot (intervention); benefit = five_shot - zero_shot, in pp
  M = 5 models, K = 4 draws

                                             benefit  acc. zero_shot  acc. five_shot
R (reliability across draws)                   0.252           0.851           0.943
  signal S (pp^2)                              22.68          197.30          307.36
  within-draw variance T (pp^2)                90.02          231.90          325.85
  prompt noise T - S (pp^2)                    67.34           34.60           18.49
C observed between distinct draws              0.700           0.875           1.000
  share of tied comparisons                    0.000           0.050           0.000
C_ref = 1/2 + arcsin(R)/pi                     0.581           0.824           0.892
R of the 4-draw mean                           0.574           0.958           0.985
C_ref for two 4-draw means                     0.695           0.907           0.945
draws for 80% agreement                           13               1               1
draws for 90% agreement                           58               4               2
status                                      resolved        resolved        resolved

Per model, mean over the 4 draws (pp)
model                        zero_shot  five_shot     gain   gain SD
Llama-3.2-1B-Instruct            31.00      33.75     2.75     10.69
Llama-3.2-3B-Instruct            57.00      61.50     4.50      5.45
Meta-Llama-3-8B-Instruct         70.00      80.50    10.50      8.96
Mistral-7B-Instruct-v0.3         52.50      71.25    18.75      6.70
OLMoE-1B-7B-0924-Instruct        46.00      56.50    10.50      5.97
```

R of benefit is 0.25. Under the reference, two evaluations on one configuration each would order a pair of these models by benefit the same way 58% of the time, a little better than a coin flip. The four draws agree on 70% of comparisons, which with ten model pairs and six pairs of draws is a coarse figure. Averaging the four pilot draws raises R to 0.57 and C_ref to 0.69. Each evaluation would need 13 draws for 80% agreement and 58 for 90%. Accuracy under zero-shot orders the same models far more repeatably (R 0.85, one draw for 80%). The standard deviation of each model's gain across configurations is 5 to 11 pp, comparable to the spread of the models' mean gains, which run from 2.75 to 18.75 pp.

The contrast here is five-shot against zero-shot, so the comparison-arm check does not apply.

## Example 2: GSM8K with 29 models

Twenty-nine open models on the 1,319 GSM8K test items. A requested direct answer is the baseline and zero-shot chain of thought the intervention, with strict scoring of the boxed answer. The four draws are the first four of the study's discovery draws from a bank of 56 configurations. The token file holds each model's 5,276 direct-answer outputs on those draws, summarized as counts per length.

```
python pilot_report.py examples/gsm8k29_scores.csv --baseline direct_answer --intervention zero_shot_cot --tokens examples/gsm8k29_tokens.csv
```

```
Pilot report (pilot_report 1.1.0)
  arms: direct_answer (baseline), zero_shot_cot (intervention); benefit = zero_shot_cot - direct_answer, in pp
  M = 29 models, K = 4 draws

                                                 benefit  acc. direct_answer  acc. zero_shot_cot
R (reliability across draws)                       0.985               0.985               0.999
  signal S (pp^2)                                 632.89              628.15              871.78
  within-draw variance T (pp^2)                   642.56              637.50              872.26
  prompt noise T - S (pp^2)                         9.67                9.34                0.48
C observed between distinct draws                  0.954               0.948               0.976
  share of tied comparisons                        0.002               0.001               0.004
C_ref = 1/2 + arcsin(R)/pi                         0.945               0.945               0.989
R of the 4-draw mean                               0.996               0.996               1.000
C_ref for two 4-draw means                         0.972               0.973               0.995
draws for 80% agreement                                1                   1                   1
draws for 90% agreement                                1                   1                   1
status                                          resolved            resolved            resolved

Per model, mean over the 4 draws (pp)
model                                  direct_answer  zero_shot_cot     gain   gain SD
gemma-4-31b-it                                 95.13          95.75     0.63      0.07
meta-llama-3-8b-instruct                       13.65          74.30    60.65      5.35
mistral-7b-instruct-v0.3                        7.37          41.26    33.89      0.22
mistral-7b-instruct-v0.1                        4.13          27.77    23.64      0.96
mistral-nemo-instruct-2407                     19.92          82.18    62.26      3.04
mistral-small-3.1-24b-instruct-2503            29.85          92.57    62.72      0.58
phi-3-mini-4k-instruct                         44.69          82.88    38.19      6.82
phi-3-medium-4k-instruct                       38.12          91.15    53.03      2.29
phi-4                                          64.14          94.83    30.69      5.02
qwen2.5-7b-instruct                            20.38          92.04    71.66      0.80
qwen3-8b                                       86.62          93.97     7.35      1.33
qwen3-30b-a3b-instruct-2507                    80.08          94.79    14.71      1.31
qwen3.6-27b                                    54.87          97.37    42.49      2.20
gemma-2b-it                                     1.31           4.25     2.94      0.27
gemma-7b-it                                    14.25          32.49    18.23      2.79
gemma-3-12b-it                                 18.16          93.04    74.89      0.97
llama-3.1-8b-instruct                          19.37          85.77    66.39      7.77
llama-2-7b-chat-hf                              3.01           9.02     6.01      0.44
llama-2-13b-chat-hf                            13.23          19.35     6.12      3.94
gemma-3-4b-it                                   6.63          87.66    81.03      1.01
gemma-2-9b-it                                  18.80          84.06    65.26      0.46
gemma-2-27b-it                                 29.00          90.58    61.58      0.26
gemma-3-27b-it                                 27.92          95.58    67.66      0.79
qwen3-30b-a3b                                  54.02          95.20    41.19      4.97
qwen3-14b                                      39.61          94.94    55.33      1.95
qwen3-32b                                      46.34          94.48    48.14      5.23
qwen3.5-9b                                     20.34          94.88    74.55      0.93
qwen3.5-35b-a3b                                40.85          96.21    55.36      0.46
qwen3.6-35b-a3b                                28.53          96.80    68.27      4.12

Comparison-arm check: direct_answer outputs longer than 16 tokens (flagged above 5%)
model                                  outputs    over   share  median    max  verdict
gemma-4-31b-it                           5,276   5,227   99.1%   113.0    780  comparison arm does not do what its name says
meta-llama-3-8b-instruct                 5,276   1,395   26.4%     8.0   1024  comparison arm does not do what its name says
mistral-7b-instruct-v0.3                 5,276   2,702   51.2%    18.0   1024  comparison arm does not do what its name says
mistral-7b-instruct-v0.1                 5,276   2,225   42.2%    15.0    205  comparison arm does not do what its name says
mistral-nemo-instruct-2407               5,276   2,173   41.2%    14.0    357  comparison arm does not do what its name says
mistral-small-3.1-24b-instruct-2503      5,276     523    9.9%     9.0    369  comparison arm does not do what its name says
phi-3-mini-4k-instruct                   5,276   4,286   81.2%    27.0    682  comparison arm does not do what its name says
phi-3-medium-4k-instruct                 5,276   4,376   82.9%    24.0    272  comparison arm does not do what its name says
phi-4                                    5,276   3,357   63.6%    22.0    477  comparison arm does not do what its name says
qwen2.5-7b-instruct                      5,276       5    0.1%     7.0    119  ok
qwen3-8b                                 5,276   4,959   94.0%   102.0   1024  comparison arm does not do what its name says
qwen3-30b-a3b-instruct-2507              5,276   4,286   81.2%    77.0   1024  comparison arm does not do what its name says
qwen3.6-27b                              5,276     334    6.3%     7.0   1024  comparison arm does not do what its name says
gemma-2b-it                              5,276   4,835   91.6%    26.0    195  comparison arm does not do what its name says
gemma-7b-it                              5,276   5,275  100.0%    62.0    302  comparison arm does not do what its name says
gemma-3-12b-it                           5,276     395    7.5%    11.0     34  comparison arm does not do what its name says
llama-3.1-8b-instruct                    5,276   1,113   21.1%     8.0   1024  comparison arm does not do what its name says
llama-2-7b-chat-hf                       5,276   5,218   98.9%    47.0    474  comparison arm does not do what its name says
llama-2-13b-chat-hf                      5,276   5,276  100.0%    62.0    462  comparison arm does not do what its name says
gemma-3-4b-it                            5,276     575   10.9%    10.0   1024  comparison arm does not do what its name says
gemma-2-9b-it                            5,276   1,416   26.8%    14.0     45  comparison arm does not do what its name says
gemma-2-27b-it                           5,276      62    1.2%     8.0     27  ok
gemma-3-27b-it                           5,276     402    7.6%    11.0     29  comparison arm does not do what its name says
qwen3-30b-a3b                            5,276   2,925   55.4%    19.0   1024  comparison arm does not do what its name says
qwen3-14b                                5,276     695   13.2%     8.0    202  comparison arm does not do what its name says
qwen3-32b                                5,276     774   14.7%     7.0    346  comparison arm does not do what its name says
qwen3.5-9b                               5,276     242    4.6%    10.0   1024  ok
qwen3.5-35b-a3b                          5,276       5    0.1%     7.0     87  ok
qwen3.6-35b-a3b                          5,276      78    1.5%    10.0   1024  ok
  flagged: 24 of 29 models (gemma-4-31b-it, meta-llama-3-8b-instruct, mistral-7b-instruct-v0.3,
    mistral-7b-instruct-v0.1, mistral-nemo-instruct-2407, mistral-small-3.1-24b-instruct-2503,
    phi-3-mini-4k-instruct, phi-3-medium-4k-instruct, phi-4, qwen3-8b, qwen3-30b-a3b-instruct-2507,
    qwen3.6-27b, gemma-2b-it, gemma-7b-it, gemma-3-12b-it, llama-3.1-8b-instruct, llama-2-7b-chat-hf,
    llama-2-13b-chat-hf, gemma-3-4b-it, gemma-2-9b-it, gemma-3-27b-it, qwen3-30b-a3b, qwen3-14b, qwen3-32b)
```

Benefit orderings repeat on this panel: R is 0.985, the four draws agree on 95% of comparisons, and one draw already reaches 90% under the reference. The check is where this panel needs attention. Twenty-four of the 29 models write more than 16 tokens on more than 5% of items when asked for a direct answer. The median direct answer of gemma-4-31b-it, qwen3-8b and qwen3-30b-a3b-instruct-2507 runs to 77 to 113 tokens, and their small gains (0.63, 7.35 and 14.71 pp) set chain of thought against reasoning these models did unasked. For gemma-3-27b-it (median 11 and maximum 29 tokens) the flag comes from a sentence around the answer. The gains repeat across prompts, yet for the flagged models part of what they measure is how far each model reasons when told to answer directly.

## Example 3: GSM8K under both baselines

The panel, items and four draws of Example 2, with the forced baseline added (arm `forced_direct`): the opening of the boxed answer is written into the reply and the model may add at most 16 tokens. Scoring is strict. The token file is Example 2's, so w is the share of each model's requested replies longer than 16 tokens. The capability score is the Epoch Capabilities Index (Epoch AI, snapshot of 30 October 2026). The item file holds the split pilot's scores, 1 or 0 under each of the three arms, for 100 items per draw: for each draw, the items of the first of the study's saved 100-item pilots on that draw, whose items were drawn at random from one half of the 1,319 items.

```
python pilot_report.py examples/gsm8k29_two_baselines_scores.csv --baseline direct_answer --forced forced_direct --intervention zero_shot_cot --tokens examples/gsm8k29_tokens.csv --capability examples/gsm8k29_capability.csv --items examples/gsm8k29_pilot_items.csv --decision
```

```
Pilot report (pilot_report 1.1.0)
  arms: direct_answer (requested baseline, req), forced_direct (forced baseline, forced), zero_shot_cot (intervention, interv.)
  benefit = zero_shot_cot - baseline, in pp, under each baseline
  M = 29 models, K = 4 draws

                                         benefit/req  benefit/forced        acc. req     acc. forced    acc. interv.
R (reliability across draws)                   0.985           0.999           0.985           1.000           0.999
  signal S (pp^2)                             632.89          422.20          628.15          267.14          871.78
  within-draw variance T (pp^2)               642.56          422.74          637.50          267.26          872.26
  prompt noise T - S (pp^2)                     9.67            0.53            9.34            0.11            0.48
C observed between distinct draws              0.954           0.991           0.948           0.996           0.976
  share of tied comparisons                    0.002           0.000           0.001           0.001           0.004
C_ref = 1/2 + arcsin(R)/pi                     0.945           0.984           0.945           0.991           0.989
R of the 4-draw mean                           0.996           1.000           0.996           1.000           1.000
C_ref for two 4-draw means                     0.972           0.992           0.973           0.995           0.995
draws for 80% agreement                            1               1               1               1               1
draws for 90% agreement                            1               1               1               1               1
status                                      resolved        resolved        resolved        resolved        resolved

Per model, mean over the 4 draws (pp)
  req = direct_answer, forced = forced_direct, interv. = zero_shot_cot; gain/req = interv. - req, gain/forced
    = interv. - forced, SD its standard deviation across draws; difference = gain/req - gain/forced = forced -
    req; w = share of direct_answer replies longer than 16 tokens
model                                     req  forced  interv.  gain/req     SD  gain/forced     SD  difference       w
gemma-4-31b-it                          95.13   67.38    95.75      0.63   0.07        28.37   0.27      -27.75   99.1%
meta-llama-3-8b-instruct                13.65   11.39    74.30     60.65   5.35        62.91   0.60       -2.26   26.4%
mistral-7b-instruct-v0.3                 7.37    6.94    41.26     33.89   0.22        34.33   0.34       -0.44   51.2%
mistral-7b-instruct-v0.1                 4.13    5.71    27.77     23.64   0.96        22.06   0.96        1.57   42.2%
mistral-nemo-instruct-2407              19.92   14.39    82.18     62.26   3.04        67.80   0.60       -5.53   41.2%
mistral-small-3.1-24b-instruct-2503     29.85   29.17    92.57     62.72   0.58        63.40   0.24       -0.68    9.9%
phi-3-mini-4k-instruct                  44.69   18.18    82.88     38.19   6.82        64.71   0.40      -26.52   81.2%
phi-3-medium-4k-instruct                38.12   27.46    91.15     53.03   2.29        63.68   0.26      -10.65   82.9%
phi-4                                   64.14   35.65    94.83     30.69   5.02        59.17   0.47      -28.49   63.6%
qwen2.5-7b-instruct                     20.38   23.60    92.04     71.66   0.80        68.44   0.23        3.22    0.1%
qwen3-8b                                86.62   25.74    93.97      7.35   1.33        68.23   0.38      -60.88   94.0%
qwen3-30b-a3b-instruct-2507             80.08   41.36    94.79     14.71   1.31        53.43   0.75      -38.72   81.2%
qwen3.6-27b                             54.87   50.30    97.37     42.49   2.20        47.06   0.46       -4.57    6.3%
gemma-2b-it                              1.31    4.19     4.25      2.94   0.27         0.06   0.74        2.88   91.6%
gemma-7b-it                             14.25    2.79    32.49     18.23   2.79        29.70   0.89      -11.47  100.0%
gemma-3-12b-it                          18.16   20.00    93.04     74.89   0.97        73.05   0.90        1.84    7.5%
llama-3.1-8b-instruct                   19.37   15.13    85.77     66.39   7.77        70.64   1.04       -4.25   21.1%
llama-2-7b-chat-hf                       3.01    1.95     9.02      6.01   0.44         7.07   0.45       -1.06   98.9%
llama-2-13b-chat-hf                     13.23    3.47    19.35      6.12   3.94        15.88   1.99       -9.76  100.0%
gemma-3-4b-it                            6.63    9.63    87.66     81.03   1.01        78.03   1.62        2.99   10.9%
gemma-2-9b-it                           18.80   21.47    84.06     65.26   0.46        62.59   1.00        2.67   26.8%
gemma-2-27b-it                          29.00   29.32    90.58     61.58   0.26        61.26   0.44        0.32    1.2%
gemma-3-27b-it                          27.92   31.33    95.58     67.66   0.79        64.25   0.16        3.41    7.6%
qwen3-30b-a3b                           54.02   37.21    95.20     41.19   4.97        58.00   0.14      -16.81   55.4%
qwen3-14b                               39.61   36.22    94.94     55.33   1.95        58.72   0.36       -3.39   13.2%
qwen3-32b                               46.34   40.58    94.48     48.14   5.23        53.90   0.30       -5.76   14.7%
qwen3.5-9b                              20.34   28.07    94.88     74.55   0.93        66.81   0.30        7.73    4.6%
qwen3.5-35b-a3b                         40.85   41.17    96.21     55.36   0.46        55.04   0.16        0.32    0.1%
qwen3.6-35b-a3b                         28.53   46.32    96.80     68.27   4.12        50.47   0.52       17.80    1.5%
  w, mean over the 29 models with token counts: 42.6%

Comparison-arm check: direct_answer outputs longer than 16 tokens (flagged above 5%)
model                                  outputs    over   share  median    max  verdict
gemma-4-31b-it                           5,276   5,227   99.1%   113.0    780  comparison arm does not do what its name says
meta-llama-3-8b-instruct                 5,276   1,395   26.4%     8.0   1024  comparison arm does not do what its name says
mistral-7b-instruct-v0.3                 5,276   2,702   51.2%    18.0   1024  comparison arm does not do what its name says
mistral-7b-instruct-v0.1                 5,276   2,225   42.2%    15.0    205  comparison arm does not do what its name says
mistral-nemo-instruct-2407               5,276   2,173   41.2%    14.0    357  comparison arm does not do what its name says
mistral-small-3.1-24b-instruct-2503      5,276     523    9.9%     9.0    369  comparison arm does not do what its name says
phi-3-mini-4k-instruct                   5,276   4,286   81.2%    27.0    682  comparison arm does not do what its name says
phi-3-medium-4k-instruct                 5,276   4,376   82.9%    24.0    272  comparison arm does not do what its name says
phi-4                                    5,276   3,357   63.6%    22.0    477  comparison arm does not do what its name says
qwen2.5-7b-instruct                      5,276       5    0.1%     7.0    119  ok
qwen3-8b                                 5,276   4,959   94.0%   102.0   1024  comparison arm does not do what its name says
qwen3-30b-a3b-instruct-2507              5,276   4,286   81.2%    77.0   1024  comparison arm does not do what its name says
qwen3.6-27b                              5,276     334    6.3%     7.0   1024  comparison arm does not do what its name says
gemma-2b-it                              5,276   4,835   91.6%    26.0    195  comparison arm does not do what its name says
gemma-7b-it                              5,276   5,275  100.0%    62.0    302  comparison arm does not do what its name says
gemma-3-12b-it                           5,276     395    7.5%    11.0     34  comparison arm does not do what its name says
llama-3.1-8b-instruct                    5,276   1,113   21.1%     8.0   1024  comparison arm does not do what its name says
llama-2-7b-chat-hf                       5,276   5,218   98.9%    47.0    474  comparison arm does not do what its name says
llama-2-13b-chat-hf                      5,276   5,276  100.0%    62.0    462  comparison arm does not do what its name says
gemma-3-4b-it                            5,276     575   10.9%    10.0   1024  comparison arm does not do what its name says
gemma-2-9b-it                            5,276   1,416   26.8%    14.0     45  comparison arm does not do what its name says
gemma-2-27b-it                           5,276      62    1.2%     8.0     27  ok
gemma-3-27b-it                           5,276     402    7.6%    11.0     29  comparison arm does not do what its name says
qwen3-30b-a3b                            5,276   2,925   55.4%    19.0   1024  comparison arm does not do what its name says
qwen3-14b                                5,276     695   13.2%     8.0    202  comparison arm does not do what its name says
qwen3-32b                                5,276     774   14.7%     7.0    346  comparison arm does not do what its name says
qwen3.5-9b                               5,276     242    4.6%    10.0   1024  ok
qwen3.5-35b-a3b                          5,276       5    0.1%     7.0     87  ok
qwen3.6-35b-a3b                          5,276      78    1.5%    10.0   1024  ok
  flagged: 24 of 29 models (gemma-4-31b-it, meta-llama-3-8b-instruct, mistral-7b-instruct-v0.3,
    mistral-7b-instruct-v0.1, mistral-nemo-instruct-2407, mistral-small-3.1-24b-instruct-2503,
    phi-3-mini-4k-instruct, phi-3-medium-4k-instruct, phi-4, qwen3-8b, qwen3-30b-a3b-instruct-2507,
    qwen3.6-27b, gemma-2b-it, gemma-7b-it, gemma-3-12b-it, llama-3.1-8b-instruct, llama-2-7b-chat-hf,
    llama-2-13b-chat-hf, gemma-3-4b-it, gemma-2-9b-it, gemma-3-27b-it, qwen3-30b-a3b, qwen3-14b, qwen3-32b)

Who benefits (gain = zero_shot_cot - baseline; mean over the 4 draws)
                                                        requested       forced
capability slope on own-baseline accuracy (pp/pp)          -0.310        0.341
  A, cross-draw moment of gain and accuracy (pp^2)        -194.63        91.22
  V_x, cross-draw variance of accuracy (pp^2)              628.15       267.14
slope on the capability score (29 models)                  0.4370       0.6767
  V_x, variance of the capability score                   197.826      197.826
top third by mean gain (10 of 29 models)
  requested: mistral-nemo-instruct-2407, mistral-small-3.1-24b-instruct-2503, qwen2.5-7b-instruct,
    gemma-3-12b-it, llama-3.1-8b-instruct, gemma-3-4b-it, gemma-2-9b-it, gemma-3-27b-it, qwen3.5-9b,
    qwen3.6-35b-a3b
  forced: mistral-nemo-instruct-2407, phi-3-mini-4k-instruct, phi-3-medium-4k-instruct, qwen2.5-7b-instruct,
    qwen3-8b, gemma-3-12b-it, llama-3.1-8b-instruct, gemma-3-4b-it, gemma-3-27b-it, qwen3.5-9b
  shared by the two top thirds: 7 of 10
model pairs the two baselines order oppositely by gain: 82 of 406 (20.2%); pairs tied under either: 0
  with both gaps above 1 pp: 70 of 384; above 2 pp: 60 of 363; above 5 pp: 35 of 296; above 10 pp: 12 of 213

Split pilot (item-level scores: 4 draws, 100 items per draw)
  assignment: half of each draw's items to each baseline by numpy.random.default_rng(20261121), the same for
    every model; 50 items per draw on the requested baseline
model                                  gain/req  gain/forced  split gain  chosen
gemma-4-31b-it                             1.25        28.00       13.25        
meta-llama-3-8b-instruct                  59.50        63.25       63.50     yes
mistral-7b-instruct-v0.3                  37.50        38.00       37.50        
mistral-7b-instruct-v0.1                  26.00        25.50       26.25        
mistral-nemo-instruct-2407                66.25        73.75       71.00     yes
mistral-small-3.1-24b-instruct-2503       62.25        62.75       62.75     yes
phi-3-mini-4k-instruct                    39.25        67.00       53.25        
phi-3-medium-4k-instruct                  52.75        67.25       58.00        
phi-4                                     28.75        61.00       43.75        
qwen2.5-7b-instruct                       76.75        73.00       75.00     yes
qwen3-8b                                   7.75        69.25       39.25        
qwen3-30b-a3b-instruct-2507               15.50        56.00       35.25        
qwen3.6-27b                               46.25        50.50       47.50        
gemma-2b-it                                2.75         0.75        3.25        
gemma-7b-it                               17.00        31.75       26.25        
gemma-3-12b-it                            79.50        77.25       78.50     yes
llama-3.1-8b-instruct                     65.00        72.25       71.25     yes
llama-2-7b-chat-hf                         5.00         5.75        5.00        
llama-2-13b-chat-hf                        0.25        13.75        7.50        
gemma-3-4b-it                             80.25        78.75       80.00     yes
gemma-2-9b-it                             71.00        68.25       70.50     yes
gemma-2-27b-it                            63.00        62.00       62.25        
gemma-3-27b-it                            69.75        66.25       68.50     yes
qwen3-30b-a3b                             43.75        59.75       51.25        
qwen3-14b                                 60.25        62.50       61.00        
qwen3-32b                                 48.50        56.00       53.00        
qwen3.5-9b                                77.50        69.50       74.00     yes
qwen3.5-35b-a3b                           57.25        56.25       56.75        
qwen3.6-35b-a3b                           69.50        49.50       57.50        
  gain/req and gain/forced: the same items, all under one baseline (100 x mean paired gain)
  top third by split gain (10 of 29): meta-llama-3-8b-instruct, mistral-nemo-instruct-2407,
    mistral-small-3.1-24b-instruct-2503, qwen2.5-7b-instruct, gemma-3-12b-it, llama-3.1-8b-instruct,
    gemma-3-4b-it, gemma-2-9b-it, gemma-3-27b-it, qwen3.5-9b
  shared with the requested third of the draw-level gains: 9 of 10; with the forced third of the draw-level
    gains: 7 of 10
  shared with the third the same items give under one baseline: requested: 8 of 10; forced: 7 of 10

Decision: zero_shot_cot for the top third (10 of 29 models)
                                                           requested          forced
own-gain third and capability third share                    3 of 10         1 of 10
split-half regret, own gain (pp per chosen model)               0.34            0.00
  smallest to largest over the splits                   0.14 to 0.60    0.00 to 0.03
split-half regret, capability slope                            32.50           15.72
  smallest to largest over the splits                 31.88 to 33.04  15.48 to 15.96
  splits: 6 choices (every split of the 4 draws into halves of 2 and 2, both directions); the regret is the
    best third's mean gain minus the chosen third's, on the half not used to choose
  requested, third by own mean gain: mistral-nemo-instruct-2407, mistral-small-3.1-24b-instruct-2503,
    qwen2.5-7b-instruct, gemma-3-12b-it, llama-3.1-8b-instruct, gemma-3-4b-it, gemma-2-9b-it, gemma-3-27b-it,
    qwen3.5-9b, qwen3.6-35b-a3b
  requested, third by the capability slope: meta-llama-3-8b-instruct, mistral-7b-instruct-v0.3,
    mistral-7b-instruct-v0.1, gemma-2b-it, gemma-7b-it, gemma-3-12b-it, llama-2-7b-chat-hf,
    llama-2-13b-chat-hf, gemma-3-4b-it, gemma-2-9b-it
  forced, third by own mean gain: mistral-nemo-instruct-2407, phi-3-mini-4k-instruct,
    phi-3-medium-4k-instruct, qwen2.5-7b-instruct, qwen3-8b, gemma-3-12b-it, llama-3.1-8b-instruct,
    gemma-3-4b-it, gemma-3-27b-it, qwen3.5-9b
  forced, third by the capability slope: gemma-4-31b-it, phi-4, qwen3-30b-a3b-instruct-2507, qwen3.6-27b,
    gemma-3-27b-it, qwen3-30b-a3b, qwen3-14b, qwen3-32b, qwen3.5-35b-a3b, qwen3.6-35b-a3b
```

Benefit repeats across draws under both baselines (R 0.985 over the requested baseline, 0.999 over the forced one). The baselines differ most for the models that reason when asked for a direct answer. Qwen3-8B writes more than 16 tokens in 94% of its requested replies, reaches 86.62% accuracy that way against 25.74% when forced, and its gain from chain of thought is 7.35 pp over the requested baseline and 68.23 pp over the forced one. Across the panel w averages 42.6%.

The claim of who benefits depends on the baseline. The capability slope on own-baseline accuracy is −0.31 under the requested baseline, so models with lower accuracy gain more, and 0.34 under the forced baseline, so models with higher accuracy gain more. On the capability index, which places every model on one axis, both slopes are positive (0.44 and 0.68 pp per index point). The two top thirds share 7 of their 10 models, and the two baselines order 82 of the 406 model pairs (20.2%) oppositely, 12 of them with gaps above 10 pp under both. On the study's 24 discovery draws the same quantities are −0.320, 0.343 and 83 of 406.

The split pilot, with 50 of each draw's 100 items under each baseline, chooses a third that shares 9 models with the requested baseline's third and 7 with the forced baseline's. In the decision block, own mean gain chooses close to the best third under either baseline (split-half regret 0.34 and 0.00 pp per chosen model). The capability slope chooses the models of lowest requested accuracy under one baseline and of highest forced accuracy under the other, sharing 3 and 1 of 10 models with the own-gain thirds, and its regret is 32.50 and 15.72 pp.

## Rebuilding the examples

`make_examples.py` writes the files in `examples/` from the study's processed data, which is not public at this stage. From the root of the code release, with those data restored there:

```
python pilot_report/make_examples.py
```

- `dove_arc_challenge_scores.csv`: the first benchmark in the fixed order of `manifests/sampler_reliability_map_dove_tasks.json` (alphabetical by DOVE file name), with that manifest's four pilot configurations.
- `gsm8k29_scores.csv`: the first four `discovery_config_ids` of `manifests/sampler_gsm8k.json`, arms `direct_answer` and `zero_shot_cot`, strict scoring.
- `gsm8k29_tokens.csv`: the `direct_answer` outputs of the same draws, as counts per (model, arm, length).
- `gsm8k29_two_baselines_scores.csv`: the same draws and arms with `forced_direct` added, strict scoring; the builder checks that the requested and chain-of-thought scores equal `gsm8k29_scores.csv` and that all three arms equal the study's cached arrays of its discovery draws.
- `gsm8k29_pilot_items.csv`: one row per model, draw and item, with one column per arm (strict scores, 1 or 0). The rule for the items: for each of the four draws, the 100 items of the first saved pilot on that draw in the study's item-disjoint 100-item pilots (`manifests/predictions_decision_strict.npz`), in that pilot's order. The builder checks that the gains over each baseline on these items reproduce the pilots' saved predictions.
- `gsm8k29_capability.csv`: the Epoch Capabilities Index of each model, as the study reads it. Its first line is the source credit, written as a comment: `# Epoch Capabilities Index, Epoch AI, CC BY 4.0, retrieved 30 October 2026, https://epoch.ai/eci`.

The score files carry an extra column, `config_id`, naming the configuration behind each draw; the script ignores it.

## How the numbers are computed

X is an array of draws by models in percentage points: benefit G = P − B, or one arm's accuracy.

- Each draw is centred over models: X̃[k, m] = X[k, m] − mean over m of X[k, m].
- S = Σ over ordered pairs of distinct draws k ≠ l, and over models m, of X̃[k, m] X̃[l, m], divided by K (K − 1) (M − 1).
- T = the mean over draws of Σ over m of X̃[k, m]², divided by M − 1.
- R = S / T, undefined when T = 0.
- C: for a model pair (i, j), let p and n count the draws in which X[k, i] − X[k, j] is above 1e-9 and below −1e-9. Of the K (K − 1) ordered pairs of distinct draws, (p (p − 1) + n (n − 1)) order the pair the same way, 2 p n order it oppositely, and the rest involve a tie. C = same + tie / 2, averaged over model pairs with equal weights.
- C_ref(k) = 1/2 + arcsin(R_k)/π with R_k = k R / (1 + (k − 1) R), defined for R in [0, 1].
- Draws for a target agreement a: max(1, ⌈s (1 − R) / (R (1 − s))⌉) with s = sin(π (a − 1/2)), the smallest k with C_ref(k) ≥ a. None when R is not finite or R ≤ 0; one when R = 1.
- w: the number of the model's requested outputs longer than the token budget, divided by the number of its requested outputs.
- Capability slope of gain G on x: A = Σ over ordered pairs of distinct draws k ≠ l, and over models m, of G̃[k, m] x̃[l, m], divided by K (K − 1) (M − 1); V_x is the same sum with x in place of G; slope = A / V_x, undefined when V_x ≤ 0. x is the baseline's accuracy, or the capability score repeated in every draw (V_x is then the variance of the score across models, with denominator M − 1).
- Top third: n = round(M/3), at least 1 and at most M − 1. With v the n-th largest value, the models above v + 1e-9 are chosen, and the models within 1e-9 of v share the remaining places equally. Two thirds with weights w₁ and w₂ share Σ over m of min(w₁[m], w₂[m]) models.
- Opposite orderings: for models i and j, d₁ and d₂ are the differences of their mean gains under the two baselines. The pair is tied when |d₁| ≤ 1e-9 or |d₂| ≤ 1e-9, and opposite when it is not tied and d₁ and d₂ differ in sign. It counts above a threshold t when min(|d₁|, |d₂|) > t + 1e-9.
- Split pilot: the draws are taken in order of first appearance, and each draw's n items in file order (with a `stratum` column, grouped by stratum, strata in order of first appearance). U = numpy.random.default_rng(seed).random((D, S, n/S)) for D draws and S strata; within each stratum, the ⌊k/2⌋ of its k items with the lowest argsort ranks of U go to the requested baseline and the rest to the forced baseline. Split gain = 100 × the mean over all the pilot's items of (intervention score − score under the item's baseline).
- Capability third: prediction = mean gain over models + slope × (the model's mean accuracy − the mean over models), with the slope from all draws. An undefined slope gives every model the same prediction, so all tie.
- Split-half regret: for a choice half C and an evaluation half E, with w the third chosen from the draws in C and ḡ the mean gains over the draws in E, regret = (Σ over m of w*[m] ḡ[m] − Σ over m of w[m] ḡ[m]) / n, where w* is the top third by ḡ. The mean is over every split into halves of ⌊K/2⌋ and ⌈K/2⌉ draws, in both directions, when there are at most 20,000 such choices; otherwise over 10,000 random splits (seed `--seed`), in both directions.

These are the estimators of the study's analysis code. The study's tests check that the script reproduces them to 1e-12 on random arrays and on the three examples.

## Credits

The capability scores in `examples/gsm8k29_capability.csv` are from the Epoch Capabilities Index (Epoch AI, CC BY 4.0, retrieved 30 October 2026, https://epoch.ai/eci).
