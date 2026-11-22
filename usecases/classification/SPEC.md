# Classification use case

## Formulation
- One row per Snapshot, all 24 Channels: "Which workload class dominates the trace?" Prompt and 12 options come from `finetune_kernelsight.py` (read from the file). Answer: the class name (at most 4 tokens). `max_new_tokens` 16.
- Why: the class-only run (3 epochs, 1 run, test .897 / .717 / .692 acc / balanced / macro-F1) is the reference. Identical rows keep the multi-task arm comparable.

## Rows
train 1,509 rows (862 Snapshots, 235 Problems); validation 291 (82 Problems); test 291 (84 Problems). Gemma 3 tokenizer: question + answer 101 tokens; with the 24 Channel lines 882.

## Oversampling stays
Train repeats each rare class to 100 rows (matmul keeps 509), as the reference did. A change would confound "mix vs class-only". In a larger mix, softmax (7 Snapshots) would almost vanish. Cost: about 4 min per epoch. `flamingo/multitask.py` uses all 1,509 rows each epoch and does not subsample them. `--min-per-class 0` builds a no-repeat ablation.

## Metric and scorer
`score.py` (numpy). `main` = balanced accuracy; also accuracy, macro-F1, per-class F1, Problem-weighted accuracy, invalid count. An output that names no option is wrong. Self-test: answers 1.0; random .117 (chance .083). `cluster_interval` resamples Problems; `paired_difference` gives a − b on the same draws.

## Baselines (test, 1 run each, 95% Problem-cluster interval)
| model | acc | balanced | F1 |
|---|---|---|---|
| majority | .584 | .091 | .067 |
| text-only LR | .911 | .751 [.703, .963] | .711 |
| MiniRocket seed 0 | .948 | .826 [.800, .993] | .849 |
| CNN seed 0 | .928 | .778 [.744, .979] | .760 |
| Flamingo class-only | .897 | .717 [.637, .936] | .692 |
| 4 mix pilots, 1 epoch | .84–.91 | .655–.776 | .60–.76 |

Text-only LR reads mean and std as the prompt prints them. Its predictions equal LR on raw values: the text line alone carries the full LR score. Flamingo − text LR −.034 [−.154, +.059]; MiniRocket − Flamingo +.109 [+.022, +.197]. Validation text LR: .900 / .716.

## Controls
The class counts Bins, so `reversed` = `shuffled` = the answer.
- **Swapped (main).** Donor: `donors.json`, a fixed uniform draw (seed 0, with replacement) from the same split, of another class (so another Problem). A derangement is impossible: 170 of 291 test Snapshots are matmul. The encoder gets the donor's scaled series; each text line keeps the own mean and std. Do not use `own_mean + own_std * z(donor)`: it fails in the 14% of Channel pairs where exactly one is constant. Check: prompt = own prompt; values = donor values. `swap_read`: "own" above `chance_own` = text reliance; "donor" above `chance_donor` = series reliance.
- **Shuffled.** The text line stays the same. An equal score = no use of Bin order (expected: MiniRocket trained on shuffled Bins keeps .774). It cannot separate text from series. A drop = order-dependent features or an input the encoder never saw.
- **Reversed.** Same reading, weaker. Optional.

## Expected outcome
- Class score within about ±.06 of class-only. Evidence: 4 mix pilots − class-only = −.014 [−.100, +.044].
- Swap: "own" near the real accuracy, "donor" near chance. The LLM agrees with text LR on 88% of test rows, but MiniRocket does too (92%). Only the swap decides.
- No model beats MiniRocket.

## Go/no-go on validation
Needs real and swapped validation outputs, and a class-only rerun as reference (runs/flamingo has no validation outputs; its model is gone). Without a rerun, the reference is text LR (.716).
- NO-GO if invalid > 15 of 291, fewer than 8 of 10 classes predicted, or balanced < reference − .05. Then class-only stays the class headline and the multi-task score is reported as "diluted".
- Claim "reads the series" only if the test donor-share interval is above `chance_donor`.

## GPU time share
Training about 9.7 min per epoch (2.6 rows/s). About 2 min per 291-row pass (estimated from file times). Full: 6 passes (validation real, swapped; test real, swapped, shuffled, reversed), about 12 min. With 2 epochs, about 31 min. A class-only rerun adds about 40 min.

## Risks
A larger mix than the pilots (43% class rows) can dilute. Validation has no softmax. Four test classes have 1 Problem each (one Problem moves balanced accuracy by .09). Softmax repeats 14 times (overfit). The swap needs a default-off utils hook (`flamingo/multitask.py`). One run per recipe.

## Round-1 minimum
All class rows in the mix. Passes: validation real; test real, swapped, shuffled (about 8 min). Compare with .717, labelled "one run".

## Sample rows (prompt and options shortened)
```
{"id": "classification/train/000929", "task": "classification", "type": "workload_class", "path": "kernels/kernelbench/_out/L1_P24/input/tensor_input.npz", "channels": null, "prompt": "These are the 24 hardware counters ... Answer with one of the answer_options.", "options": ["matmul", "conv", ..., "other"], "answer": "softmax", "controls": {"reversed": "softmax", "shuffled": "softmax"}, "max_new_tokens": 16}
{"id": "classification/validation/000203", "path": "kernels/kernelbench/_out/L1_P33/dt-bf16_s123/input/tensor_input.npz", "answer": "normalization", ...}
{"id": "classification/test/000000", "path": "kernels/cutlass_fmha/_out/b2h16_q2048_d128_causal/input/tensor_input.npz", "answer": "attention", ...}
```

## Commands
```
uv run usecases/classification/build.py               # rows + donors.json, 3 s
uv run --with numpy usecases/classification/score.py  # self-test, 1 s
uv run usecases/classification/baselines.py           # baselines.json, 1 min
```
