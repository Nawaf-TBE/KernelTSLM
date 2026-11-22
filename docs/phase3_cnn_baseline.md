# Phase 3 — trained, evaluated and frozen L1 CNN baseline

19 November 2026 (Asia/Muscat). Both predeclared experiments are complete.
**Selected baseline: unweighted CNN, epoch 7, validation macro-F1
0.945925 over the fixed 11 training-supported classes.**
The selected checkpoint and configuration are copied read-only under
`outputs/phase3/cnn/frozen/`, with SHA256 verification and independent prediction
replay. No test evaluation, QA generation, L2 training or OpenTSLM training ran.

## Representation and architecture

Input is only normalized float32 **[B,24,512] counter X**. The model is never
passed labels, validity masks, timestamps, metadata, filenames, fingerprints or
class text. Targets/masks move separately to the device solely for loss/metrics.
All 512 original bins remain input context, including unknown primary bins.
The 12 original output channels/IDs remain in order. Output is **[B,12,512]**.

The network has **126,540 trainable parameters**:

1. Conv1d 24→64, kernel 1.
2. Three width-64 residual blocks with dilations 1,2,4. Each block has **two**
   temporal convolutions: channel-only LayerNorm → GELU → Conv1d(kernel 5,
   dilation d, padding 2d) → dropout 0.1, repeated twice, followed by identity
   residual addition. LayerNorm normalizes 64 channels independently per timestep;
   it does not mix temporal or cross-trace statistics.
3. Per-time channel LayerNorm → GELU → Conv1d 64→12, kernel 1.

All strides are 1. Each convolution is centered at its original input bin, with
same-length zero padding; no crop, interpolation or resampling occurs. The
maximum effective receptive field is calculated, not inferred from names:

`R = 1 + (5-1) × 2 × (1+2+4) = 57 bins`, i.e. 28 bins on either side away from
edges. Pointwise projections/normalizations/activations do not expand it.
A gradient-support test and shift-equivariance test verify this alignment. Since
bin durations differ by trace, 57 bins is not asserted to be a fixed duration in
nanoseconds. The model sees the complete trace, but a timestep has local context.

## Data access and normalization

The unchanged Phase 3 pipeline supplies the physically isolated train/validation
TimeF views. Only those two releases can be opened by the training reader and
row-group cache guards. Raw NPZ array decoding is blocked in model development.
No original mixed-partition TimeF reader is opened; preservation hashing reads
file bytes only and never deserializes its test-containing row groups.
All **1,124 training** and **160 validation** traces are used every epoch without
trace/bin subsampling or validation oversampling. Official paths/partitions are
unchanged. Batches shuffle whole training traces; validation order is official.

To avoid repeated TimeF JSON decoding, an in-RAM cache holds unchanged X, primary
L1 labels, valid masks and evaluation bookkeeping from `make_dataloaders`.
`collate_traces` and `valid_primary_logits` are reused. This cache occupies
65.83 MiB of tensor buffers; its train/validation contents
and identity orders have fingerprints. It is not a new split or feature modality.
No auxiliary supervision is passed to the CNN. Per-bin flattening is not used by
this temporal model; all chronological bins remain intact.

The verified Phase 1 training-only population z-score is reused, unchanged:
`66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`. Its population includes all 575,488 training bins,
including inactive and unlabeled context. No validation normalization is fitted.
Raw/compiled inputs, official splits, old outputs, old code and both isolated
views are protected by before/after SHA256 inventories.

## Incomplete labels and controlled class weighting

There are **233,821 valid L1 training bins / 575,488 total (40.6300%)** and
**341,667 unknown (-1) bins**. Unknown means unknown, not idle/background/other.
The loss selects exactly `valid_l1`, computes per-bin cross-entropy, sums it and
divides by the **number of valid bins**, including for the weighted condition.
Unknown logits have zero direct classification-loss gradient, although unknown
counter context may influence neighboring valid predictions through convolutions.
No pseudo-labels are used. A completely unsupervised batch returns `None` and
skips the entire optimizer step (including AdamW weight decay). Unit tests verify
this; no such all-empty batch occurred in these complete training epochs.

Experiment A uses unweighted masked cross-entropy. Experiment B computes weights
**only from valid training-bin counts**. For each positive-support class:

`r_c = min(4, sqrt(max_training_count / count_c))`

`w_c = r_c / (sum(count_c × r_c) / N_valid)`

Zero-support `other` has weight 0 for positive training targets. Positive weight
max/min is at most 4, and the training-bin-weighted mean is one (float32 roundoff
recorded in `class_weights.json`). The softmax still includes all 12 competitors,
so zero positive support does not remove output 11. Validation loss/metrics use
unweighted cross-entropy for fair reporting. Curves also show the weighted
training objective separately from unweighted train/validation CE.

Matmul has **122,535 / 233,821 = 52.4055%** of valid training bins. No examples
were invented for `other`, which has **zero** positive training examples and is
not claimed as a learned class. Weighting is tested, not presumed beneficial.

## Fixed protocol, sanity and determinism

`configs/phase3_cnn.json` and `outputs/phase3/cnn/protocol.json` predeclare seed 7,
batch 16, AdamW, LR 1e-3, weight decay 1e-4, at most 30 epochs, patience 5,
constant learning rate and strict-improvement selection by validation macro-F1
on IDs 0–10. No supported IDs are dropped when absent in a diagnostic subset.
Within a run, ties retain the earliest epoch; across conditions, exact ties choose
unweighted. No diagnostic metric is used for selection. Both conditions have
identical initial parameter hashes and matching training trace-order hashes for
all common epochs. Sanity overfit weights are discarded; each experiment starts
from the same fresh seeded initialization.

Before full training: **9 CNN unit tests passed**, covering shapes, exact receptive
field/alignment, manual CE, weighted denominator, unknown gradients, zero-label
updates, class weights, confusion metrics, diagnostics and CPU/MPS backward.
MPS then passed a 10-update actual-counter stability test and CPU-evaluation
parity (maximum absolute logit difference 1.43e-6). A deterministic training-only
subset of the first two multiclass traces with >=32 valid bins reached **98.26%
accuracy after 25 updates**, with CE 0.3823, meeting the predeclared >=95% accuracy
and <=20% initial-loss criteria. This sanity fit does not contribute final metrics.
An initial API-name test failure was fixed before any full experiment; its logs
are retained under `sanity_attempt1_api_fix/`.

Python, NumPy and PyTorch/MPS seeds are set. Deterministic algorithms are requested
with `warn_only=True`, but **MPS reports no deterministic implementation for
`index_put_with_accumulate_mps` during masked backward**. Thus identical seeds/order
are a controlled comparison, not a bitwise training reproducibility guarantee.
Forward replay of the frozen model matches all saved probabilities within
rtol 1e-5 / atol 1e-6. CPU/MPS/backend/version differences may change retraining.
No automatic CPU fallback was needed; actual training used Apple MPS.

## Validation results

All **33,416 valid validation bins** are scored, out of 81,920 (40.7910% coverage).
**48,504 unknown validation bins** are excluded. These are 160 trace units, not
33,416 independent samples. There are no validation traces with zero valid bins;
the general rule excludes such traces from per-trace means and records their
count. Accuracy/micro-F1 use valid bins only. Macro-F1 is an unweighted class mean.
All-12 macro-F1 uses `zero_division=0`, assigning F1=0 to a class with neither truth
nor predictions; its explicit unsupported `other` entry is retained.

| Metric | Unweighted | Weighted |
| --- | ---: | ---: |
| Selected epoch | 7 | 13 |
| Epochs executed | 12 | 18 |
| Macro-F1, fixed supported 11 | 0.945925 | 0.943411 |
| Macro-F1, all 12 | 0.867098 | 0.864794 |
| Masked accuracy = micro-F1 | 0.959331 | 0.958762 |
| Mean per-trace accuracy | 0.949780 | 0.951529 |
| Mean per-trace macro-F1, fixed 11 | 0.090638 | 0.090494 |
| Mean per-trace macro-F1, truth-present classes | 0.928606 | 0.930120 |
| Best-checkpoint training macro-F1 | 0.950056 | 0.965189 |

Per-trace fixed-11 F1 is low because most traces contain only one or few classes:
absent-class F1 is zero by definition. The truth-present per-trace average is
reported separately to make that convention clear; neither selects checkpoints.

| ID | Class | Validation support | Unweighted precision | Unweighted recall | Unweighted F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | matmul | 15,644 | 0.9994 | 0.9829 | 0.9911 | 0.9910 |
| 1 | conv | 2,322 | 0.8316 | 0.9083 | 0.8683 | 0.8619 |
| 2 | activation | 2,675 | 0.8997 | 0.9492 | 0.9238 | 0.9264 |
| 3 | normalization | 1,591 | 1.0000 | 0.9635 | 0.9814 | 0.9572 |
| 4 | softmax | 439 | 1.0000 | 0.9932 | 0.9966 | 0.9966 |
| 5 | pooling | 1,400 | 0.9091 | 0.9500 | 0.9291 | 0.9030 |
| 6 | reduction | 3,308 | 0.9981 | 0.9441 | 0.9703 | 0.9714 |
| 7 | attention | 584 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | loss | 1,113 | 0.9991 | 0.9793 | 0.9891 | 0.9987 |
| 9 | elementwise | 1,794 | 0.7365 | 0.7776 | 0.7565 | 0.7714 |
| 10 | memory_movement | 2,546 | 1.0000 | 0.9980 | 0.9990 | 1.0000 |
| 11 | other | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Full precision/recall/F1/support for the weighted run are also saved in
`weighted/per_class_metrics.csv` and `weighted/validation_results.json`.

Weighting changes macro-F1 by **-0.002514**
(-0.2514 percentage points) in this pair.
It improves some class F1s (notably loss and elementwise) but loses more on
normalization/pooling and does not improve the selection metric. With one seed
and MPS nondeterminism, this small difference is not a statistical claim that
weighting is generally harmful. The unweighted checkpoint wins the declared rule.

Minority-class recognition is meaningful within labeled validation coverage:
e.g. softmax has F1 0.9966 on 439 bins, loss 0.9891 on 1,113, and attention 1.0 on
584. These bin counts are not independent trace replications or evidence of
universal GPU-operation recognition. Elementwise (0.7565) and conv (0.8683) remain
weaker. Neither model meets the predeclared >=95% single-class prediction-collapse
flag. Unweighted predicts matmul for **46.0468%** of valid validation bins, versus
**46.8159%** true matmul support; this is not matmul collapse.

The unweighted model's online training CE continues down after epoch 7 while
validation F1 no longer improves (last epoch F1 0.9341), consistent with later
training not improving held-out selection performance. Its best-checkpoint
train/val F1 gap is only 0.0041. The weighted best gap is 0.0218; late weighted
train and validation scores both deteriorate, so the decline cannot be attributed
solely to overfitting rather than optimization variability. Both stopped after
five epochs without a new best, not by selecting diagnostic results.

## Predeclared temporal diagnostics (validation only)

A well-annotated *interior* is a contiguous valid constant-primary-class run of
length >=9, excluding four bins at each edge. Unknown gaps break runs.
A class *boundary* exists only between two adjacent valid bins with different
classes; score four bins on each side, intersecting valid labels. Unknown↔known
edges are not operation changes. These definitions were saved before results.
A multiclass trace has >=2 distinct valid L1 IDs; score all its valid bins.
There are **13** such validation traces. Diagnostics are not new selection metrics.

| Subset | Valid bins | Unweighted accuracy | Unweighted fixed-11 F1 | Weighted accuracy | Weighted fixed-11 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| interior | 20,821 | 0.9998 | 0.9048 | 1.0000 | 0.9091 |
| boundary | 2,234 | 0.6392 | 0.2452 | 0.6012 | 0.2481 |
| multiclass_traces | 2,752 | 0.6944 | 0.2449 | 0.6650 | 0.2509 |

Interior accuracy is excellent, while boundary and multiclass-trace accuracy are
substantially weaker. Fixed-11 F1 remains subject to subset class coverage:
interiors have no softmax truth, so even perfect predictions for the ten present
classes yield at most 10/11; boundaries/multiclass subsets contain eight supported
truth classes. This does not justify dropping absent classes after seeing results.
The tables and full per-class diagnostic JSON preserve the predeclared convention.
Common errors include elementwise→activation (232 bins), matmul→conv (229),
elementwise→conv (161), reduction→elementwise (135), and conv→pooling (133).
Counts are not causal explanations. Source labels describe device-marker envelopes,
not instruction-level simultaneity. No overlap score is derived from single-label
predictions, and no L2/multi-label/explanation-quality result is claimed.

## Runtime, resources and artifacts

Hardware: Apple MPS on arm64 macOS 26.3.1, **24 GiB physical system RAM**; CPU
compatibility/parity used four PyTorch threads. Python 3.13.4, PyTorch 2.13.0,
NumPy 2.5.0, TimeNet 0.12.0; exact versions/source/view/cache hashes are saved.
Unweighted timed training plus final evaluations took **23.49s**;
weighted took **33.45s**. Total timed work after initial
unit checks (including TimeF cache load, device/overfit sanity and preservation)
was **87.44s**. Fresh plotting/replay/audits are additional.
Observed full-process peak RSS was **4,045,504,512 bytes
(3.768 GiB)**. This is not the earlier 3 GiB conversion budget;
that threshold applied to the conversion task, not a training memory guarantee.
Sampled MPS driver allocations peaked at 0.119 / 0.134 GiB in the two runs;
these are after-batch observations, not a complete GPU allocation high-water.
Do not add them to RSS as independent memory on this unified-memory machine.
No new package was installed and the original Phase 1 environment is unchanged.

Artifacts under `outputs/phase3/cnn/`:

- `protocol.json`, `provenance.json`, `class_weights.json`, source preservation
  inventories and exact selected configuration.
- `sanity/`: test XML/log, actual CPU/MPS check and tiny overfit history.
- `unweighted/`, `weighted/`: `best.pt`, `history.json`, `results.json`,
  `validation_results.json`, per-class/per-trace CSV, raw confusion CSV, confusion
  PNG, learning curves PNG, prediction distribution PNG and validation predictions.
- `validation_predictions.npz` schema: `trace_ids` maps trace indices to original
  paths; each remaining row has `trace_index`, original `bin_index`, `true_class`,
  `predicted_class` and float32 `[12]` probabilities. There is exactly one row for
  every valid bin and no rows for unknowns. Vocab order is in provenance.
- `per_class_comparison.png`, `summary.json`, `artifact_verification.json`.
- `frozen/baseline.pt`, `frozen/config.json`, `frozen/manifest.json`: read-only
  selected baseline and hashes, suitable for future OpenTSLM comparison.
  `frozen/normalization.json` is an exact statistics copy; `frozen/source/` archives
  the executed model/training/configuration and unchanged data helpers.
  `reproduction_manifest.json` and `bundle.json` pin these files by hash.

Selected checkpoint SHA256:
`2e960150ef4820c038684e33102c0892c930c451f5d6904aa99aae542855e5d5`.
Independent artifact checks recompute both confusion matrices/F1s from predictions,
verify every trace/bin/target against the isolated validation cache, verify best
epochs and frozen hashes, and replay all frozen-model probabilities on MPS.
The combined regression suite passed **57 tests**, with **one skip** for unavailable CUDA.
This includes all earlier loader/pipeline tests and CPU/MPS CNN checks.
All earlier protected files remain unchanged. No original dataset/split/view is
rewritten; no model checkpoint from another task is overwritten.

## Reproduce and scope of conclusions

From the Kernel X-ray root, using the existing TimeNet environment:

```bash
.worktrees/timenet-kernelsight/.venv/bin/uv run \
  --project .worktrees/timenet-kernelsight python scripts/run_cnn_baselines.py
.worktrees/timenet-kernelsight/.venv/bin/uv run \
  --project .worktrees/timenet-kernelsight python scripts/verify_cnn_artifacts.py
.venv/bin/python scripts/report_cnn_baselines.py
```

The training runner deliberately refuses existing experiment/sanity/frozen folders.
To repeat training, use a separate clean checkout/output copy with the same
config and isolated views; preserve the existing frozen run. The supplied output
already contains the trained artifacts and must not be deleted to rerun. MPS
requires execution permissions unavailable in some app sandboxes. No full mixed
TimeF or test loader should be used. Report rendering uses the unchanged Phase 1
Matplotlib environment and was visually checked.

This establishes **primary L1 recognition on the annotated portion of the official
validation partition**, with promising minority-class results but weaker temporal
transition/composition behavior. It is not L2 classification, overlapping-kernel
detection, explanation quality, or test performance. Trace path disjointness does
not prove kernel/workload-identity independence. Missing labels, class imbalance,
unsupported `other`, annotation granularity, single-seed variability and repeated
validation selection limit generalization claims. Macro/per-class F1 is more
informative than accuracy alone; valid bin count is not independent trace count.
The frozen baseline is retained for later authorized comparison. No automatic QA,
OpenTSLM, L2 or test-stage work follows this task.
