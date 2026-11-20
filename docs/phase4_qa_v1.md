# Phase 4 — Mathematically verifiable QA benchmark v1

## Purpose and scope

This benchmark tests three mathematical operations on measured GPU counter time
series: interval means, maximum comparisons, and unique-peak ordering. Reference
answers come from deterministic arithmetic, never an LLM or CNN predictions.
It prepares supervision for later OpenTSLM experiments; **no OpenTSLM training or
QA accuracy is reported**. Generated benchmark data remains local and unpublished.

The official split is preserved: 1,124 training and 160 validation traces, with
24 channels and 512 bins each. No official test trace, counter row group, label,
or model tensor is decoded. Path-disjoint traces are not necessarily independent
workload identities, as the [CNN audit](phase3_cnn_independent_audit.md) established.

## Representation and isolation

The existing `TimeFTraceDataset` reads the physically separate train/validation
views. Its existing identity checks account for physical row groups and reject
the original mixed release. `isolated_reads()` guards reader initialization,
row-group reads, and raw NPZ loads throughout generation and verification.
The QA dataset factory rejects `test` before constructing a reader.
The underlying verified adapter validates label consistency internally, but QA
uses only `X`, counter names, and trace/split bookkeeping. No L1/L2 labels,
activity/supervision masks, segments, fingerprints, or CNN outputs select or answer
questions. All 512 bins are eligible, including bins without operation labels.

`X` is the unchanged normalized **float32 [24,512]** tensor. Original channel and
temporal order, counter names, and original trace identities are retained.
The saved Phase 1 statistics are fitted exclusively on all 575,488 training bins;
there is no new fitting on training or validation. Subtraction/division use the
loader's float64 calculation followed by float32 casting. The saved statistics
SHA256 is `66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`.

**Answers are in normalized units, not physical hardware units.** Comparing two
channel-wise z-scores compares relative training-distribution excursions, not
absolute utilization, throughput, or causality. Temporal answers refer to original
bin indices, not interpolated continuous-time peak locations. Native timestamp
rounding is irrelevant to these bin-index questions. The average weights bins
equally; it is not a duration-weighted physical integral.

## Oracle definitions

All intervals are nonempty, half-open `[start,end)` within `[0,512]`. The
[oracle](../src/kernel_xray/qa/oracle.py) rejects malformed shapes, non-float32 or
nonfinite tensors, invalid channel indices, repeated pair indices, invalid name
mappings, and invalid intervals. It promotes the selected float32 values to
float64 for arithmetic. Constants are fixed in
[`configs/phase4_qa_v1.json`](../configs/phase4_qa_v1.json), saved in `protocol.json`
before final generation, and checked for equality on verification.

1. **Statistics:** `sum(X[c,start:end])/N`, with float64 accumulation and `N=end-start`.
   The reference includes sum and count. Future numerical answers are accepted
   when `abs(pred-ref) <= 1e-6 + 1e-5*abs(ref)`; references themselves retain full
   JSON float precision. NaN/infinite answers are invalid.
2. **Comparison:** compute both maxima. Return `Counter A`, `Counter B`, or
   `Tie` for exact equality. Reject nonzero differences at or below
   `1e-4 + 1e-3*max(abs(maxA),abs(maxB))`. Generation also rejects exact ties to
   favor clear differences; the oracle and tests retain explicit Tie semantics.
   Operands record both maxima, gap, and separation threshold.
3. **Temporal:** for each counter, require exactly one peak and a top-minus-
   runner-up gap strictly greater than `1e-6 + 1e-5*abs(peak)`. Repeated or
   numerically ambiguous maxima are rejected. Return `Yes`, `No`, or `Same bin`
   from original peak indices. Operands include maxima, runner-ups, gaps,
   thresholds, and original peak bins. `argmax` is used only after uniqueness is
   established; its first-tie behavior never defines a temporal answer.

## Sampling and category balance

Seed **7** is hashed with split and trace identity to give stable per-trace RNGs;
these seeds/identities never enter model inputs. Traverse traces in official
split order. Target five questions per type per trace. Choose an interval band
with probability 1/3: lengths **16–64**, **65–192**, or **193–512** inclusive.
Choose length uniformly within the band, then a valid start uniformly. Choose
one or two distinct counters uniformly without replacement, with random pair
orientation. No operation-label condition is consulted.

For statistics, retain five valid distinct candidates. For each categorical type,
sample a fixed pool of up to **1,500 attempts**, deduplicating exact parameter
questions and rejecting mathematical ambiguity. Select from the least represented
available answer category per trace, breaking selection ties randomly. Never
reverse an incorrect answer or force an invalid example to fill a quota.
Pools can exhaust; recorded shortfalls mean insufficient valid sampled candidates,
not a proof that no valid interval exists anywhere in the trace.
This creates approximately balanced sampled categories, not natural prevalence.
Accepted interval/counter distributions can differ from proposal distributions.

## Generated dataset and quality

| Split | Statistics | Comparison | Temporal | Total |
|---|---:|---:|---:|---:|
| Train | 5,620 | 5,620 | 5,412 | 16,652 |
| Validation | 800 | 800 | 769 | 2,369 |
| Total | 6,420 | 6,420 | 6,181 | **19,021** |

All targets were met on 1,054/1,124 training traces (**93.77%**) and 149/160
validation traces (**93.125%**). All shortfalls are temporal: 208 missing training
and 31 missing validation examples. Repeated/ambiguous peaks dominate rejections:
1,552,690 training and 221,560 validation candidate rejections. Comparison near-tie
rejections are 1,199/174; duplicate candidates 58/9. No invalid questions were
forced. Normalized means range from approximately -0.802039 to 11.411753 in
training and -0.802039 to 5.685437 in validation.

| Categories | Train | Validation |
|---|---|---|
| Comparison A / B / Tie | 2,839 / 2,781 / 0 | 409 / 391 / 0 |
| Temporal Yes / No / Same bin | 1,733 / 1,715 / 1,964 | 243 / 237 / 289 |

`quality.json` includes per-trace shortfalls/reasons, questions-per-trace counts,
complete interval-length histograms, channel-index frequency histograms, numerical
ranges, and answer distributions. Full channel names are in each example and
model-facing tensor metadata. These distributions describe selected examples,
not annotation coverage of the original dataset.

### Shortcut diagnostics

A template-only majority rule learned on training answers obtains **51.125%**
comparison and **37.581%** temporal validation accuracy. A training-derived ordered
counter-pair majority lookup (fallback: training global majority) obtains
**56.125%** comparison and **61.769%** temporal accuracy. This exposes substantial
counter-identity/synchrony shortcuts, particularly for temporal questions.
These are diagnostic no-time-series lookup rules, not OpenTSLM results. They do
not use validation answers to fit the rule. The limited fixed templates also
make task identity easy to infer, though wording does not encode the oracle answer.
Future model evaluations should include these controls; accuracy above random
chance alone does not establish temporal reasoning. No v1 thresholds or splits
were tuned after observing this validation diagnostic.

## Records and safe model inputs

JSONL files are `outputs/phase4/qa_v1/train.jsonl` and `val.jsonl`.
Each record contains a SHA256 question ID, official split/trace ID, question type,
channel indices/names, start/end, question text, answer/type, oracle intermediates
and version/precision, and normalization path/hash. Dataset hashes and view-manifest
hashes are recorded in verification/protocol files. No counter data is duplicated
in each JSONL record; it is resolved through the authorized trace loader.

`model_input(example, X, names)` in
[`generation.py`](../src/kernel_xray/qa/generation.py) returns exactly:

```python
{"question": "...", "time_series": X.tolist(), "counter_names": names}
```

The full original-order normalized trace is supplied, not an answer-bearing
summary or selected maxima. Python JSON numbers round-trip to the same float32
values; future adapters must cast to float32 and preserve order and all input
precision. No ID, path, split, oracle operands, answer, label, mask, or fingerprint
is serialized. Never pass the full QA record as a prompt. Tests mutate answer and
oracle fields and assert identical model inputs. Future evaluation should report
numeric tolerance accuracy, exact categorical accuracy, per-type coverage, and
trace-level uncertainty without tuning on the official test partition.

## Independent verification and tests

[`verification.py`](../src/kernel_xray/qa/verification.py) does not import/call the
oracle. It recomputes sums with `math.fsum`, maxima with Python `max`, runner-ups
with scalar sorting, and unique peak locations by explicit enumeration. It
checks schema, text, identity hash, split, channel names/order, numeric/category
answers, and all saved operands. Operands must agree within a tighter `1e-12`
relative/absolute scale; numerical answers use the declared evaluation tolerance.
The build driver also checks unique question identities/text, official membership,
no split overlap, serializer allowlist/float32 round-trip, persisted byte hashes,
and exact regeneration under the frozen configuration. Any invariant failure
raises and writes a failed verification status.

Generation checks each example and regenerates each trace a second time. The
separate `--verify` command reopens authorized views, parses saved JSONL, regenerates
its exact bytes, and independently verifies every saved answer. No earlier CNN
metric helper is used as a QA reference. Preservation compares original byte
hashes for prior data, views, outputs, source, tests, configuration, and docs;
new Phase 4 implementation files are separately versioned as development artifacts.

The 33 focused synthetic tests cover means, all category outcomes, ties/near-ties,
unique/repeated/ambiguous peaks, interval bounds, nonfinite/malformed inputs,
mapping validation, deduplication, repeatability, split rejection, leakage
sentinels, independent verification, and corrupted question rejection.
Full project regression: **94 passed, one CUDA-availability skip**; no failures.

## Reproduction commands

Use the existing prepared workspace; the data/views/checkpoint are not in GitHub.
The Phase 3 TimeNet environment uses Python 3.13; root `.venv` is retained solely
for compatible visualization. No new dependencies are required.

```sh
export PYTHONPATH="$PWD/src:$PWD"
export UV_CACHE_DIR="$PWD/.cache/uv"
UV="$PWD/.worktrees/timenet-kernelsight/.venv/bin/uv"
"$UV" run --project .worktrees/timenet-kernelsight python scripts/build_qa_v1.py
"$UV" run --project .worktrees/timenet-kernelsight python scripts/build_qa_v1.py --verify
"$UV" run --project .worktrees/timenet-kernelsight python -m pytest tests -q -rs --junitxml=outputs/phase4/qa_v1/regression.xml
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python scripts/plot_qa_v1.py
"$UV" run --project .worktrees/timenet-kernelsight python scripts/finalize_qa_v1.py
```

Generation overwrites only the Phase 4 QA artifacts under the same frozen
protocol. Keep `preservation_before.json`; do not replace it to make checks pass.
Do not invoke historical verifiers that overwrite their original evidence.
Tests involving host MPS/shared memory may need host execution permissions.
No command above trains a model, decodes official test arrays, commits, pushes,
or publishes generated data. Changes remain local pending separate review.

## Figures and real examples

Four figures are under `outputs/phase4/qa_v1/figures/`: a three-panel distribution
summary and one real **training-only trace per question type**. Each example
shows the selected interval and full trace, counter names, mean/max values,
and reference answer in normalized units. `illustrations.json` retains their
exact measured normalized tensors and question records for reproduction.

## Limitations and next steps

Flat/inactive channels and repeated plateaus are deliberately excluded from
unique-peak temporal QA, creating selection bias. Comparison can be answered from
baseline z-scores for some inactive intervals; it need not involve active GPU
execution. No operation semantics, concurrent-kernel detection, L2 recognition,
causal explanation, or unseen-hardware generalization is established. Trace-
level bootstrap or grouped workload analyses will be needed for later model
comparisons. The QA types involve exact mathematical operators and narrow fixed
templates, not broad linguistic comprehension. Future benchmark versions can
study counter-identity controls, active/flat strata, plateau-aware questions,
physical-unit alternatives, richer language, duration-aware integration, and
separate operation/overlap explanations, with new preregistered protocols.


### Example questions and answers (training only)

- **statistics**: What is the average value of sass_control between time bins 249 and 275? Use [249, 275) in normalized units.
  Reference: `0.2745972596682035`. Trace: `kernels/cutlass_fmha/_out/b1h32/input/tensor_input.npz`.
- **comparison**: Which counter has the higher maximum value between bins [15, 80): pred_on_per_inst_ratio or stall: barrier? Values are normalized.
  Reference: `Counter B`. Trace: `kernels/cutlass_fmha/_out/b1h32_q4096_d128_causal/input/tensor_input.npz`.
- **temporal**: Does hit: l2 reach its maximum before pipe: lsu within bins [26, 511)? Values are normalized.
  Reference: `Yes`. Trace: `kernels/cutlass_fmha/_out/b1h32_q4096_d128_no/input/tensor_input.npz`.
