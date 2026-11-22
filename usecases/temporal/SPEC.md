# Temporal reasoning (TMP)

## Formulation and why
Each row shows ONE Channel, and the answer is a function of that series alone. The question reads only the last series (utils.py:196). In pilots 2-3c that series was Tensor Active, which is 0 for most non-tensor kernels. Answers are closed options. Pilot 1 collapsed to the letter "C", and 23-29 of its 30 free-number answers were invalid.

| Type | Channel | Options | Evidence |
|---|---|---|---|
| trend | named | increasing / decreasing | reversal pairs; train balanced per Channel |
| iteration_at_bin | sass_control | 1-3 / 4-11 / 12-22 / 23 or more | only pilot order use: 23 vs 0, z 4.8 |
| longest_gap_len | sass_control | 1-4 / 5-6 / 7-21 / 22 or more | 14 vs 6 |
| peak_side | named | before / after Bin N | N balanced per 32-Bin range |

`sass_control > 0` equals the busy mask on all Bins. Dropped types:
- peak_order, onset: two series; the model collapsed to "same time".
- kernel_edge: 10 vs 5; 70% of answers do not change on reversal.
- loop_period: collapsed to one answer.
- bin_state, class_switch, sequence types: 82-100% same answers under shuffle; SEG covers per-Bin classes.

## Rows (mean tokens 156-164, Gemma 3 tokenizer; class row 1,050)
| | trend | peak | iter | gap | total |
|---|---|---|---|---|---|
| train | 480 | 560 | 1000 | 405 | 2,445 |
| train_reversed | 480 | – | 713 | 405 | 1,598 |
| validation | 348 | 106 | 473 | 152 | 1,079 |
| test | 306 | 112 | 447 | 135 | 1,000 |

- Caps: 40 rows per (type, Problem) in train, 10 in validation and test.
- Test trend: 306 pairs, 55 Problems, 42 forward-decreasing.
- **train_reversed.jsonl is opt-in.** Its rows carry `"input": "reversed"`, and RUN must feed x[:, ::-1] for them. If RUN cannot, leave the file out: ignored, it teaches opposite answers for one input.
- peak_side has no reversed copies: 95% of its reversed answers are "after", which leaks N.
- controls.shuffled uses `default_rng(0).permutation(512)`, as `finetune_kernelsight.shuffled` does.

## Scorer (score.py)
- `score()`: accuracy per type; main = mean over types.
- `pairs(rows, real, reversed, hard_ids)`: the share of pairs with both answers right. This is the headline. Results come with Problem-cluster intervals.
- `paired(rows, real, shuffled)`: discordant counts and Problem-clustered z.
- Self-test: answers 1.0; random .377 (chance .375); order-blind pairs 0.

## Baselines (test, baselines.json)
| | trend | peak | iter | gap | pairs trend / hard |
|---|---|---|---|---|---|
| majority | .137 | .500 | .443 | .215 | 0 |
| text-only | .137 | .500 | .602 | .215 | 0 |
| order-blind GBM | .850 | .795 | .770 | .444 | 0 / 0 |
| order-aware GBM | .944 | .848 | .895 | .733 | .935 / .696 |
| arrow + order-blind | | | | | .850 / 0 |

- The traces are centred, so the busy share gives position. **Real accuracy does not show order use. Only the controls can.**
- The arrow shortcut gets .952 on forward-decreasing pairs.
- So trend reading needs the 46 hard pairs (23 Problems), where the order-blind model is wrong.
- Order-aware real − shuffled: z_problem 4.1-10.8.

## Controls
- reversed: pair score (trend 306, peak 56, iteration 21 eligible rows).
- shuffled: run on ALL rows and score against the real answers, as in the pilots.

## Expected outcome
- iteration_at_bin: real > shuffled will likely replicate, near the .77 order-blind accuracy.
- trend: never run. After 12 of 13 negative pilot types, a pair score near 0 is likely.
- One run only.

## Go/no-go (validation only)
- GO: pair score ≥ .10 with an interval that excludes 0, or z_problem ≥ 2.
- Test claims need z_problem ≥ 2.5 (Holm over 4 types).

## GPU (about 0.22 s per training row)
- Full: 4,043 rows/epoch, about 15 min; over 3 epochs plus about 20 min of controls, about 70 min.
- Round-1 minimum: trend (with reversed copies) and iteration, 1,960 rows/epoch, about 7 min; over 2 epochs plus evaluation, about 25 min.

## Risks
- RUN may not support `input`.
- 55 trend Problems give wide intervals.
- Reversed traces are not physical.

## Samples
```
trend     DRAM Write Bandwidth: "...trend increasing or decreasing?" -> decreasing (rev increasing)
peak_side DRAM Write Bandwidth: "...highest value before or after Bin 223?" -> before (rev after)
iteration sass_control: "...which busy period contains Bin 351?" -> 1-3 (rev 1-3, shuf null)
```
