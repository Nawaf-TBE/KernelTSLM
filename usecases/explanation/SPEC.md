# Explanation: KernelSight rationales and HAR-CoT faithfulness

## Part A: KernelSight explanation rows

**Formulation.** Free text, `options: []`. The prompt asks for checked facts, then `Answer: <class>`, and lists the 12 classes. The answer is a template rationale of computed facts. All 24 Channels are used, with `SMs Active [Throughput %]` moved to the end. Reason: a text token reads only the most recent series (utils.py:196). SMs Active is > 0 in 100% of busy Bins and 0% of idle Bins, so the rationale can read the busy timeline.

**Facts (one sentence form each).**
- text: `<Channel> mean is in the top|bottom k% of training.` (percentile among the 862 training Snapshots; 2 class-typical Channels)
- series: `The GPU is busy in <bucket> of Bins.`
- order: `The GPU has N busy periods.`, `A busy period starts every P Bins.`, `SMs Active rises|falls during busy time.`, `Workload order: A -> B.`, `After A comes B.`

**Replace or add:** ADD. The classification task keeps the plain class rows as the primary class metric. There is one explanation row per Snapshot, with no repeats.

**Rows:** train 862, validation 291, test 291. The test rows are 200 plain (1-3 busy periods), 75 loop and 16 multi-workload. Gemma 3 tokens: prompt ≈ 1,009, answer mean 69 (max 107). `max_new_tokens` is 112.

**Scorer.** `score.py` parses each sentence before `Answer:`. It checks each claim against the facts of the input as given (original, reversed or shuffled; `facts.json`). `main` is the mean claim precision per row. A row with no valid `Answer:` scores 0. The scorer also reports class accuracy, precision per family, the order-claim share and the unparsable share. It reads the condition from `row["condition"]` if `flamingo/multitask.py` sets it. If not, it infers the condition from the reference text (verified: 134/134 reversed, 291/291 shuffled).

**Baselines** (test; main / order precision):

| | original | reversed (134) | shuffled |
|---|---|---|---|
| reference (self-test) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / – |
| true reference, input order ignored | 1.0 / 1.0 | .787 / .515 | .636 / .047 |
| nearest training Snapshot's rationale | .825 / .766 | .649 / .432 | .536 / .031 |
| class template of MiniRocket's class | .609 / .679 | .436 / .462 | .241 / .005 |
| another test Snapshot's rationale | .200 | .254 | .139 |

**Controls.**
- shuffled (all rows): text and series claims must stay true. Order claims must disappear or describe the shuffled timeline.
- reversed: only on rows where a reference claim becomes false (test 134: 121 from the trend, 15 from the workload order).
- swapped: skipped for this task.

**Expected outcome.**
- Class accuracy near the class rows.
- High text precision, because the means are in the text.
- The pilots barely used order. So expect shuffled order precision near .05 with an order-claim share near .39. That reads as "the explanation describes order that the input does not have".

**Pre-registered rule.** The model uses order if shuffled order precision is ≥ .15 (or the order-claim share halves) AND reversed order precision is ≥ .615.

**Go/no-go (validation, real input).** GO if unparsable ≤ .10 and main ≥ .70. Otherwise, show 3 worked examples only.

**GPU share** (from 583 s per 1,509 class rows; decoding estimated at 2-2.5 s/row):
- Training ≈ 6 min per epoch.
- Validation ≈ 11 min.
- Test: real + shuffled + reversed, 716 rows ≈ 27 min.
- Total ≈ 56 min.
- **Round-1 minimum:** 100 validation rows, then test real + shuffled ≈ 26 min.

**Risks.**
- The model can check text claims from the text line alone.
- The reversed control is mostly one fact. SMs Active falls in 605 Snapshots and rises in only 45.
- Only 16 multi-workload test Snapshots.
- `flamingo/multitask.py` must shuffle with `default_rng(0).permutation(512)`, as `finetune_kernelsight.shuffled` does.
- A rationale before the answer can change the class.
- One run.

**Samples** (prompt and channels elided):
```
plain/attention: "sass_memory_global mean is in the bottom 10% of training. gmem_coalesce_ratio mean is in the bottom 10% of training. The GPU is busy in under 25% of Bins. The GPU has 2 busy periods. Answer: attention"
loop/matmul: "... The GPU is busy in 25-50% of Bins. The GPU has 4 busy periods. A busy period starts every 51 Bins. Answer: matmul"
multi/conv: "... The GPU has 39 busy periods. A busy period starts every 8 Bins. SMs Active rises during busy time. Workload order: conv -> normalization -> activation -> pooling. After conv comes normalization. Answer: conv"
```

## Part B: HAR-CoT, OpenTSLM/gemma-3-270m-pt-har-flamingo

**Setup.** `har/run_har.py` uses OpenTSLM's HARCoTQADataset formatting, greedy decoding, batch 1 and the authors' `Answer:` parser. It writes nothing into the OpenTSLM repo.
- Windows: 400 test windows, 50 per class, from the 8,222 valid windows (CSV sha256 c8da9bcc…).
- Conditions: real; shuffled (one permutation per window, the same for x, y and z); reversed. The mean and std text does not change.

**Output per condition.**
- Accuracy, balanced accuracy and macro-F1.
- Claim rates: affirmed (no negation cue in the sentence) and mentioned.
- Measured input periodicity (ACF ≥ .5): real .62, shuffled 0.
- Answer flips compared to real.

**Reference rationales** (mentioned / affirmed): periodic .765 / .465, order .102 / .072, spike .907 / .543, variability .986 / .832. Negation matters: the model writes "lacks clear periodic patterns".

**Finding.** OpenTSLM's uv.lock pins open-flamingo 0.0.2, which cannot build the Flamingo model. The script uses open-flamingo 2.0.1 with dependency overrides and transformers <4.58.

**Tested.** Mac MPS, 2 windows × 3 conditions: 4.4 s per window, 2/2 correct. A full run takes ≈ 1.5 h on a MacBook (MPS).
