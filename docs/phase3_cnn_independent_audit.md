# Phase 3 independent CNN baseline audit

**Verdict: VERIFIED WITH QUALIFICATIONS.** The reported scores, checkpoint,
counter-only inference, and physical train/validation isolation are independently
confirmed. Material limits remain in workload generalization, validation
uncertainty, and deterministic reproduction of MPS training. No correctness or
confirmed target-leakage defect was found. No model was retrained, no checkpoint
or split changed, and no test arrays were decoded.

The machine-readable entry point is
`outputs/phase3/cnn_audit/verification.json` (local, not published). All evidence
below is under `outputs/phase3/cnn_audit/`; historical CNN results remain under
`outputs/phase3/cnn/`. This audit diagnoses the existing selected checkpoint; it
does not select a replacement or change the reported metric definitions.

## Method and independence

The audit reads the actual live and archived source, checkpoints, prediction
arrays, training histories, official train/validation membership, trusted source
NPZs, and physically isolated TimeF views. The original mixed release is never
opened with a TimeF reader. Protected test files are only hashed as opaque bytes
for preservation, not deserialized or evaluated.

`src/kernel_xray/phase3/independent_audit.py` constructs confusion counts with
`numpy.add.at` and independently calculates TP/FP/FN, precision, recall, harmonic
F1, and pooled accuracy. It does not call `cnn_metrics.py`. Its functional CNN
uses checkpoint tensors with explicit convolution, per-time channel LayerNorm,
GELU, and residual operations; it does not call the original model's `forward`.
An additional check loads the actual module in evaluation mode and compares its
forward output with this functional implementation.

The protocol was saved before the analyses: full aligned counter signatures,
cosine >=0.995 **and** relative L2 <=0.10 for near matches, CPU probability
tolerances `rtol=1e-4, atol=5e-5`, and 10,000 whole-trace bootstrap replicates with
seed 20261119. Near-match comparisons use unchanged training-only z-scores, no
per-trace fitting or time warping. All 179,840 train/validation pairs are checked.
Metadata groupings are diagnostic covariates and never enter model inputs.
The additional seed-insensitive configuration grouping explicitly distinguishes
KernelBench seed repeats from complete path identity; it is exploratory, not a
new evaluation split or selection criterion.

## 1. Source, architecture, and checkpoint integrity

`source_integrity.json` independently verifies 30 file hashes against recorded
source/configuration/bundle manifests and 38 named checks. Live execution source
equals archived execution source. The frozen file equals `unweighted/best.pt`,
contains epoch 7 and unweighted training configuration, and has SHA256:

```
2e960150ef4820c038684e33102c0892c930c451f5d6904aa99aae542855e5d5
```

The frozen configuration SHA256 is
`b86b0056da4e2e7ca2a10cda7ad64a7416c8a23a06730a33fdd95669ebbb875f`.
Both checkpoint state dictionaries have the exact declared tensor names, shapes,
finite float32 values, and **126,540 parameters**. The actual convolutions have
two kernel-5 layers per residual block, dilations 1, 2, 4 and unit stride.
Their receptive field is `1 + 2*(5-1)*(1+2+4) = 57` bins. An input-gradient
support check independently spans bins 228–284 for output bin 256. LayerNorm
acts across channels at each time position; there are no BatchNorm running
statistics extending the receptive field.

The histories contain 12 executed unweighted epochs and 18 weighted epochs.
Strict validation macro11 improvement with patience 5 selects epochs 7 and 13,
respectively. Strict `>` retains the earliest epoch on a tie. Both experiments
use the same seed, reconstructed initial parameter hash, and common shuffle
orders; all 18 recorded epoch order hashes were reconstructed independently.
Each epoch reports all 233,821 valid training bins. Nonbest epoch model states
were not retained, so their history scores are checked for consistency and
selection arithmetic, not independently rerun by retraining.

## 2. Independent validation metrics

There are exactly **33,416** valid trace/bin identities for each experiment,
covering all original valid primary L1 positions once. No missing, duplicated,
out-of-range, or unlabeled prediction entries were found. Probability arrays are
finite, normalized, and agree with saved argmax classes. Independently reconstructed
integer 12×12 confusion matrices exactly match the original reports before any
floating-point comparison.

| Experiment | Selected epoch | Macro-F1, fixed 11 | Accuracy = micro-F1 | Macro-F1, all 12 |
|---|---:|---:|---:|---:|
| Unweighted, frozen | 7 | 0.9459247843 | 0.9593308595 | 0.8670977190 |
| Weighted | 13 | 0.9434112005 | 0.9587622696 | 0.8647936004 |

All per-class precision, recall, and F1 values also match within `1e-12`.
The fixed supported set is training-derived IDs 0–10. `other` remains output
ID 11, with zero training/validation positive support and zero F1 by the
documented zero-division convention; it is not claimed as a learned class.
The all-12 score includes that zero. Unknown `-1` bins are excluded, not
reclassified: coverage is 33,416/81,920 = **40.79%**, with 48,504 unknown bins.

| L1 class | Valid validation bins | Traces containing class | Frozen F1 |
|---|---:|---:|---:|
| matmul | 15,644 | 85 | 0.991073 |
| conv | 2,322 | 16 | 0.868259 |
| activation | 2,675 | 17 | 0.923777 |
| normalization | 1,591 | 9 | 0.981434 |
| softmax | 439 | 4 | 0.996571 |
| pooling | 1,400 | 6 | 0.929095 |
| reduction | 3,308 | 17 | 0.970328 |
| attention | 584 | 9 | 1.000000 |
| loss | 1,113 | 6 | 0.989111 |
| elementwise | 1,794 | 13 | 0.756508 |
| memory_movement | 2,546 | 5 | 0.999017 |
| other | 0 | 0 | 0 |

## 3. Actual frozen CPU inference replay

Both saved models were replayed on every one of the 160 isolated validation
traces, using independently reconstructed normalized counters in original channel
and temporal order. **Every predicted class agrees** with the saved MPS-run
validation predictions. Confusion matrices, macro-F1, and accuracy are identical.
Maximum absolute probability differences were `1.0728836e-6` unweighted and
`8.3446503e-7` weighted, within the predeclared cross-device tolerance.
No parameter changed. The actual module evaluation check also matches the
independent functional model and creates no gradients or state changes.
See `cpu_replay.json` and `training_control_checks.json`.

This confirms the frozen numerical result across CPU/MPS inference. It does not
prove deterministic retraining: the recorded MPS run enabled deterministic
algorithms with `warn_only=True` and emitted an
`index_put_with_accumulate_mps` nondeterminism warning. Only one training seed was
run; this audit introduces no new training run.

## 4. Training implementation and normalization

Actual training calls `model(moved['X'])` with `[B,24,512]` float32 counters.
Targets and masks move separately for loss; metadata, filenames, fingerprints,
TimeF task objects, and timestamps are not appended to features. Source review
and a forward hook confirm the sole tensor input. Relevant references are
`cnn_training.py:103`, `:136`, and `:212`.

`cnn.py:76` selects valid primary bins, returns `None` for no supervision, and
uses `sum(weight[y] * CE)/N_valid`, or unweighted `sum(CE)/N_valid`.
`cnn_training.py:215` skips the optimizer entirely for an empty-supervision
batch, including weight decay. Unknown positions have zero loss gradient.
`cnn.py:86` forms weights from training-only label counts:
`min(4, sqrt(max_count/count))`, then rescales their training-bin expectation to
one. The ratio cap is preserved by common rescaling; `other` gets zero positive
weight but remains a softmax competitor. This denominator is the documented
definition, rather than PyTorch's weighted-mean denominator.

Independent synthetic arithmetic checks compare CE, gradients, and a manual
first AdamW update. An optimizer step is applied only to a toy logits parameter,
never to a CNN. Empty supervision skips both updates and decay. Evaluation uses
`eval()`/`no_grad()`, preserves parameters/buffers, and does not advance training
state. Selection uses the predeclared supported macro-F1.

Training-only population mean/std were independently recomputed from all 575,488
training bins, including unknown/inactive bins, and matched the saved statistics
to `1e-12`. Provenance includes every training input hash and channel order.
Statistics SHA256:
`66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`.

A **nonblocking future-configuration defect** was identified: optimizer/cache
configuration fields are recorded, but unsupported values are not rejected while
the implementation always uses AdamW and RAM caching. Actual frozen values are
AdamW/true and agree with execution, so this does not invalidate the results.
`proposed_config_guard.patch` is a separate, unapplied proposal to reject
unsupported values before any output/data access. The original sources and
frozen source bundle are unchanged. See `training_implementation_review.json`.

## 5. Strict data isolation

The actual read path was examined and exercised, not inferred from configuration.
Each view's records, counter row groups, packed payload rows, and source inventory
refer exclusively to its official partition. All physical rows are accounted for;
there are no unused rows concealing test payloads and no externally reused payloads.

| View | Records | Counter row groups | Packed payload files | Authorized source NPZ hashes |
|---|---:|---:|---:|---:|
| train | 1,124 | 14 | 27 | 3,372 |
| val | 160 | 2 | 4 | 480 |

All 1,284 input arrays and primary label arrays were independently compared with
authorized source NPZs, bit for bit. Source inventories cover all three NPZ types
by hash; fingerprint values were not used as features or similarity inputs.
Counter files are physically separate from the original mixed release and do not
alias its counter file. Original absolute nanosecond edges match canonical data;
native microsecond conversion has at most 999 ns flooring error, against minimum
native bin widths of 200 us train/201 us validation, with exact index alignment.
Model input does not use rounded timestamps.

The development `isolated_reads()` guard checks both reader initialization and
counter row-group reads against the two approved resolved view roots and blocks
raw dataset NPZ loading. A negative control rejects an unapproved release before
IO. The audit has its own guards permitting source comparison only for official
train/validation NPZs. No test records, labels, counter row groups, or model tensors
were requested or decoded. These guards cover the inspected development entry
points; they are not a general security boundary against arbitrary new Python IO.

## 6. Duplicates and workload overlap

There are **zero exact cross-partition counter-array duplicates**, zero repeated
input-container hashes, no within-partition exact counter duplicates, no shared
trace paths, and no source-counter identity aliases. Exact checks hash raw float32
bytes and confirm equality. Path disjointness does not imply independent workloads.

All **160 validation traces / 33,416 valid bins** have collector families seen in
training (11 shared collectors; one additional training-only collector). There is
no unseen-collector validation cohort. More detailed metadata groupings give:

| Definition | Seen traces / bins | Unseen traces / bins | Seen / unseen accuracy | Seen / unseen fixed-11 F1 |
|---|---:|---:|---:|---:|
| Problem-aware: KernelBench problem, otherwise collector | 140 / 29,126 | 20 / 4,290 | 0.994266 / 0.722145 | 0.899884 / 0.230320 |
| Exact set of compiled kernel symbols | 144 / 30,285 | 16 / 3,131 | 0.986297 / 0.698499 | 0.979594 / 0.298275 |

Both cohorts in these rows meet the audit's minimum 10 traces/500 valid bins,
but have unequal class composition and correlated workload identities. Problem-
aware grouping is intentionally finer for KernelBench than library collectors;
it is not a uniform semantic workload ontology. The problem-seen cohort lacks
conv truth, so its maximum fixed-11 F1 is 10/11. Both unseen cohorts have only
eight positive truth classes. These fixed-11 means must be interpreted alongside
the complete per-class counts in `workload_overlap_analysis.json`.

A new compiled-symbol **set** can combine individually familiar symbols. Only
**two** validation traces contain any individually unseen compiled symbol—too few
for a reliable unseen-kernel conclusion. Exact complete configuration paths have
no overlap, but 32 validation traces/7,137 bins share KernelBench problem/dtype
keys with training after removing only the documented seed suffix. These are
repeated nominal problem/dtype settings across seeds, not byte-identical traces;
other hidden runtime settings are not independently established. CUTLASS `_s`
tokens are retained, since they do not mean KernelBench random seeds.

Shape grouping also exposes 70 validation traces/13,830 bins sharing leading
m/n/k shape keys. This seen-shape cohort contains only matmul, so its perfect
accuracy and 1/11 fixed macro-F1 cannot be compared directly with a mixed-class
cohort. Complete paths alone would miss these repetitions.

The fixed counter-signature thresholds find **108 near pairs**, covering
**47 validation traces / 10,960 valid bins**. Of these pairs, 94 share a collector,
105 a compiled-symbol set, and 67 a shape key. The CSV retains every pair and its
scores. This is evidence of similar measured activity, **not confirmed target
leakage or identical acquisitions**. Similarity, repeated configurations, and
shared families are reported separately. No split or checkpoint was changed.

## 7. Temporal diagnostics

Independent masks exactly reproduce the original integer diagnostic matrices.
Interiors remove four bins from both ends of valid constant-class runs of length
at least nine, never bridging unknowns. Boundaries comprise four bins on each
side of an adjacent valid class change. Multiclass traces have at least two
distinct valid primary classes; all their valid bins are scored.

| Frozen-model subset | Valid bins | Accuracy | Fixed-11 F1 |
|---|---:|---:|---:|
| Interior | 20,821 | 0.9998078863 | 0.9047678822 |
| Boundary | 2,234 | 0.6392121755 | 0.2451833568 |
| Multiclass traces, 13 traces | 2,752 | 0.6944040698 | 0.2449262062 |

Interior truth lacks softmax (maximum possible fixed-11 F1 is 10/11); the other
two subsets each contain eight truth classes. Full class distributions, weighted
model diagnostics, and confusion matrices are retained in `temporal_diagnostics.json`.

Source edges, label bin identity, native tasks, and functional convolutions show
no indexing or systematic alignment defect. A diagnostic offset sweep -4..4 uses
the same 21,059 valid positions at every offset; its boundary intersection has
only 238 bins. Unweighted offset -1 improves by only four global correct bins
and three boundary correct bins versus zero; weighted performs best at zero.
There is no consistent one-bin correction supported across models. This heavily
restricted sweep is not a replacement for the original boundary metric and is
not used to realign predictions.

Observed errors concentrate around transitions and multiclass traces. Possible
explanations include model transition errors and annotation-envelope ambiguity;
these measurements cannot determine their causal contributions or recover true
instruction-level boundaries. Device-marker envelopes do not establish
instruction-level concurrency. This single-primary-label CNN does not establish
multilabel overlap detection or counter-grounded causal explanations.

## 8. Trace-level uncertainty

Bootstrap resampling draws 160 whole traces with replacement per replicate,
rebuilds pooled confusion counts, and retains the original fixed-11 class set.
Absent resampled classes receive F1 zero. These are percentile 95% intervals:

| Experiment | Macro-F1 interval | Accuracy interval |
|---|---:|---:|
| Unweighted | [0.830571, 0.968622] | [0.934317, 0.979925] |
| Weighted | [0.824810, 0.968382] | [0.932408, 0.980638] |

In 248/10,000 replicates at least one supported class is absent: softmax 157,
pooling 17, attention 1, loss 23, memory_movement 53 (counts can overlap).
The paired weighted-minus-unweighted macro-F1 difference is -0.002514, with
interval **[-0.016193, 0.003719]** using identical resamples. There is no supported
claim of meaningful weighted/unweighted superiority; the frozen unweighted
choice remains the predefined point-score selection.

These intervals condition on the selected checkpoint and this validation cohort.
They do not cover training-seed variability, validation-based selection optimism,
or extrapolation to unseen workloads/hardware. Related workloads can remain
dependent across traces, so even a trace bootstrap may understate broader
uncertainty. The 33,416 correlated bins are not 33,416 independent samples.

## 9. Regression, preservation, and reproduction

The relevant full project suite passed **61 tests with one CUDA-availability
skip**, zero failures/errors, in 35.12 seconds. Four new independent-audit tests
cover confusion arithmetic, temporal masks, functional architecture/receptive
field, and whole-trace bootstrap behavior. The original artifact verification
script also passed; its output writes were redirected into
`cnn_audit/legacy_verification/`, preserving its historical output.

The final preservation check compares SHA256 for **22,208 pre-existing files**
against the pre-audit inventory, including raw data, official splits, Phase 1/2
artifacts, the Phase 1 environment, pipeline sources, and frozen/historical CNN
outputs. All remain unchanged. New audit files are listed and hashed in
`audit_evidence_hashes.json`. No hyperparameters were tuned, no original
measurement was overwritten, and no additional baseline or test evaluation ran.

Use the existing TimeNet development environment (not the Phase 1 Python 3.9
environment). From the project root, these commands reproduce the independent
analysis and regression evidence without invoking training:

```sh
export PYTHONPATH="$PWD/src"
export UV_CACHE_DIR="$PWD/.cache/uv"
UV="$PWD/.worktrees/timenet-kernelsight/.venv/bin/uv"
"$UV" run --project .worktrees/timenet-kernelsight python scripts/audit_cnn_baseline.py
"$UV" run --project .worktrees/timenet-kernelsight python -m pytest tests -q -rs --junitxml=outputs/phase3/cnn_audit/regression.xml
"$UV" run --project .worktrees/timenet-kernelsight python scripts/regress_cnn_audit.py
python3 scripts/finalize_cnn_audit.py
```

MPS-dependent historical regression checks require execution with host MPS access;
the independent replay itself is CPU. The checked environment uses Python 3.13.4,
NumPy 2.5.0, PyTorch 2.13.0, TimeNet 0.12.0, and Arrow 24.0.0. The analysis script
fails on violated invariants. Finalization checks the pre-audit inventory and
regression evidence before issuing the qualified verdict. These commands replace
only derived audit evidence, never the original CNN artifacts. The preserved
`preservation_before.json` is the baseline, not regenerated after running checks.

## What can be shared

Share the frozen CNN as a verified **official-validation primary-L1 recognition
baseline**: macro11 0.945925, accuracy 0.959331, 40.79% labeled-bin coverage,
126,540 parameters, 57-bin receptive field, exact CPU class replay, and the
trace-cluster intervals above. Report transition/multiclass weakness, class
support, workload overlap, and MPS training nondeterminism with those numbers.

Claims about unseen kernel families, new hardware, robust temporal segmentation,
overlap detection, explanations, or a statistically superior weighting scheme
remain preliminary or unestablished. Validation is still model-selection data;
the held-out test set remains unevaluated.
