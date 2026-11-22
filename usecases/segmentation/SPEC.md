# Segmentation: per-Bin labelling as a timeline

## Formulation
- One row per Snapshot, type `timeline`, options `[]`, `max_new_tokens` 160.
- Answer: each change on a 16-Bin grid as `<first Bin> <state>`, e.g. `0 idle; 224 attention; 288 idle`. States: idle, unlabeled, 12 L1 classes. A cell is busy when at least half of its Bins are busy. A busy cell takes its most common labelled class.
- Channels: all 24, `sass_misc` last. Answer tokens read only the last series. Alone, `sass_misc` gives busy/idle .993 on validation. The default last Channel (Tensor Active) gives .842.
- One timeline row decodes all 512 Bins. Per-window questions need 32 rows per Snapshot for the same cover.
- Grid width (validation grid oracle; Gemma 3 answer tokens):

| cell | workload acc | busy/idle | boundary F1 | tokens mean / max |
|---|---|---|---|---|
| 8 | .963 | .943 | .591 | 22 / 220 |
| **16** | .953 | .937 | .563 | 17 / 140 |
| 32 | .939 | .926 | .383 | 14 / 45 |

## Rows and tokens
862 / 291 / 291 rows. About 914 tokens per row (prompt 116, series lines 781, answer 17), the same as a class row.

## Metric (score.py)
- `main`: the share of Bins whose decoded state equals the answer's state. Answers 1.0; random cells .075; another row's answer .486.
- Per-Bin against the true labels, as temporal_baselines: workload accuracy and macro-F1 on labelled Bins, busy/idle accuracy, boundary F1 (±8 Bins), and a multi-workload subset (16 Snapshots).
- Needs `temporal_qa/labels_cache.npz`. gpu/upload.sh does not send it.

## Baselines (test; validation in baselines.json)
| | main | acc | F1 | busy/idle |
|---|---|---|---|---|
| grid oracle | 1.000 | .965 | .967 | .937 |
| majority answer | .625 | .227 | .047 | .705 |
| text only (mean/std: class + centred block) | .883 | .812 | .656 | .906 |
| LR window, 3 seeds | .943 | .880 | .720 | .937 |
| same, shuffled neighbours | .941 | .875 | .713 | .937 |
| LR busy + MiniRocket class, 1 seed | .959 | .915 | .849 | .937 |

- In macro-F1 the first three rows score .768 / .710 / .706; their accuracy is .908 / .884 / .882.
- The workload part is mostly the Snapshot class. The true dominant class on all Bins gives .976. Only 3 test answers show two classes.

## Controls
- reversed: the reversed timeline. 186 of 291 answers are symmetric. Read `main_asymmetric` (n = 105).
- shuffled: the TRUE timeline. A shuffled trace has no writable timeline. A drop shows that the model reads positions. LR window drops from .943 to .656.
- swapped: the main test. A text-only model writes its own answer. A series reader writes the donor's answer.

## Expected outcome
The pilots learned priors on localization templates. Expect valid timelines, the right class, and block lengths from the text means: main near .88, and shuffled near real.

## Go / no-go (validation)
- Go: main ≥ .907 (text only + .02) and real minus shuffled ≥ .05.
- Partial: parsed ≥ .95 and main ≥ .79. Report the format, not localization.
- No-go: otherwise. Show worked examples only.

## GPU share (estimate)
Training: about 5.5 min per epoch (0.386 s per class row). Evaluation: about 0.7 s per row (not measured). Full: 3 epochs, plus validation real/shuffled and test real/reversed/shuffled/swapped (1,746 rows), ≈ 37 min.

## Round-1 minimum
1 epoch. Validation real; test real/shuffled/swapped; reversed on the 105 asymmetric rows. About 1,270 rows, ≈ 20 min.

## Risks
- Text shortcut: the means give the busy share, and all traces are centred.
- Grid ceiling: short idle gaps are lost. Multi-workload accuracy is at most .726.
- One run. Rare-class F1 is unstable (validation oracle F1 .769).
- An "a-b" range output decodes badly.

## Sample rows (channels and prompt omitted)
```
{"id":"segmentation/train/000005","path":"kernels/cutlass_fmha/_out/b2h16_q2048_d64_causal/input/tensor_input.npz","answer":"0 idle; 224 attention; 288 idle","controls":{"reversed":"0 idle; 224 attention; 288 idle","shuffled":"0 idle; 224 attention; 288 idle"}}
{"id":"segmentation/train/000150","path":"kernels/cutlass_gemm/_out/m2048_n2048_k8192_rN_s1_b0/input/tensor_input.npz","answer":"0 idle; 112 matmul; 160 idle; 288 matmul; 400 idle","controls":{"reversed":"0 idle; 112 matmul; 224 idle; 352 matmul; 400 idle","shuffled":"0 idle; 112 matmul; 160 idle; 288 matmul; 400 idle"}}
{"id":"segmentation/train/000782","path":"kernels/kernelbench/_out/L2_P4/input/tensor_input.npz","answer":"0 idle; 64 unlabeled; 80 conv; 96 activation; 480 idle","controls":{"reversed":"0 idle; 32 activation; 416 conv; 432 unlabeled; 448 idle","shuffled":"0 idle; 64 unlabeled; 80 conv; 96 activation; 480 idle"}}
```
