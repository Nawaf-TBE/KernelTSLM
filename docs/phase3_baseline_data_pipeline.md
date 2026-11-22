# Phase 3 Task 1 — baseline data pipeline

## Scope and environment

The final pipeline reads **separate, physically isolated train and validation
TimeF views** under `outputs/phase3/task1/timef_views/{train,val}/registry/`.
They preserve the approved schema and all source fields, built only from the
explicitly permitted train/val NPZs. The original accepted complete artifact at
`outputs/phase2/task3/writer/unvalidated-registry/williamhtan/kernelsight/1.0.0/`
remains unchanged and is checksum-verified as the reference release. The views
are local partial research artifacts, not publishable releases.

During initial verification, native SDK row-group caching decoded a shared
validation/test counter group in the original artifact. No test Record, label
array or model tensor was requested or used. The earlier broad no-test-decoding
claim was corrected. After the user selected **strict physical isolation**, the
pipeline switched to independently built partition views and reran all checks.
The **final** data path decodes no test arrays at any storage level. Historical
pre-policy measurements are retained under `pre_strict_policy/` and are not the
final readiness result. No model, loss optimization, OpenTSLM adapter or
fine-tuning is implemented.

New files are confined to `src/kernel_xray/phase3/`, one new test module, a view-builder and a
verification script, this document, and `outputs/phase3/task1/`. Completed Phase 1
and Phase 2 source, reports, environments and artifacts are not edited. No new
framework or dependency is installed. The separate TimeNet environment already
contains Python 3.13.4, TimeNet/connectors 0.12.0, NumPy 2.5.0 and PyTorch 2.13.0.
The original Python 3.9.6 / PyTorch 2.6.0 / NumPy 1.26.4 Phase 1 environment stays
unchanged. TimeNet requires Python >=3.11, so do not install it into the Phase 1
venv or install the root package's older dependency pins into the TimeNet venv.
The verification script adds the existing `src/` directory to its import path.

## APIs and representation

- `TimeFTraceDataset(split='train', normalization=...)`: lazy trace units in the
  official JSON list order. The default normalization is the existing Phase 1
  statistics file; pass `None` for raw model inputs. `split='val'` is supported.
  Test access is disabled by default; a future evaluation caller must explicitly
  opt in with `allow_test=True` and supply a separately authorized, isolated test view. That API option does not authorize test-driven
  development. No test Record or test NPZ is decoded; the initial incidental bulk-decode issue is disclosed above.
- `record_to_sample(record, identity, normalization)`: typed boundary between
  counter features, dense supervision and bookkeeping. Uses the connector's
  `model_features(record)` whitelist and the exact canonical dense arrays.
- `collate_traces(samples)`: stacks tensors, retains metadata as a list, rejects
  mixed partitions, channel/vocabulary orders or normalization methods.
- `make_dataloaders(batch_size=16, seed=7, num_workers=0)`: returns `train` and
  `val` loaders. Training shuffles whole traces; validation follows official
  order. Both retain the last partial batch. There is no bin-level split.
- `flatten_labeled_bins(batch, level='l1')`: selects only valid primary bins,
  returning `[N,24]` X, int64 y, and trace/bin indices for reversal/bookkeeping.
  Empty supervision returns `[0,24]` inputs and `[0]` targets.
- `valid_primary_logits(logits, batch, level='l1')`: masks `[B,C,512]` logits
  into `[N,C]` and `[N]` targets; no loss is computed. Move logits, labels and
  masks to the same device before GPU use.

For a batch of B traces:

| Field | Shape | dtype / meaning |
| --- | --- | --- |
| `X` | `[B,24,512]` | float32; the **only model input** |
| `raw_X` | `[B,24,512]` | original float32 counter measurements |
| `time_edges_ns` | `[B,513]` | original absolute int64 ns boundaries |
| `targets.workload_l1` / `workload_l2` | `[B,512]` | int64; **-1 remains unknown** |
| `targets.workload_l1_multihot` | `[B,512,12]` | float32 0/1 membership |
| `targets.workload_l2_multihot` | `[B,512,73]` | float32 0/1 membership |
| `masks.valid_l1` / `valid_l2` | `[B,512]` | bool; corresponding primary ID >=0 |
| `masks.multihot_l1` / `multihot_l2` | `[B,512]` | bool; observed positive membership |
| `masks.mask_labeled` / `mask_any_kernel` | `[B,512]` | distinct original source masks, bool |
| `targets.multihot_n_active` | `[B,512]` | int64 active L1 class count |
| `targets.overlap_l1` | `[B,512]` | bool; active L1 count >=2 |
| `targets.multihot_has_overlap` | `[B]` | bool; any multi-class overlap in trace |
| `masks.overlap_l1_valid` | `[B,512]` | bool; observed L1 membership, described below |
| `metadata` | B dictionaries | identity/split/vocabulary/order bookkeeping only |

All 12 L1 and 73 L2 names and L2 parent IDs retain their original order, including
unobserved primary classes. Labels are not remapped. The dataset checks primary
membership in multi-hot tracks, hierarchy alignment, source supervision masks,
active counts, overlap flags, original ns edges, native floor-us offsets for
every Signal, and the final right boundary. TimeNet native time remains rounded
down to microseconds; the original ns tensor is exact. Bin durations vary between
traces; no resampling or temporal reordering is introduced.

Before decoding values, the dataset requires exactly one native task partition,
exactly its official Record count, no external reused counter payloads, and
complete referenced-row coverage of every Parquet counter file and row group.
An original mixed-partition release is rejected before loading arrays. Native
Record decoding and row-group-cache guards exercise this in tests. Compiled view
hashes and selected-source NPZ hashes are recorded in each `view.json`; no test
source is opened by the view builder. Record IDs are **view-local**: use the
original `trace_id` and split for cross-view evaluation bookkeeping.

The index queries native task/input reference tables for the selected partition,
then reads identity fields for those Record IDs only. It restores official list
order by verified path and split index; it does not assume opaque Record IDs
encode source order. A read-only schema-2 DuckDB query is used only for the compact
identity index. JSON identity extraction is bounded to 16 IDs and uses one
multi-path parse per metadata value. Actual Records and Signals load through
`TimeFReader.get_many(Record, ids, load=...)`, whose API returns requested order.
No native task objects are expanded to recover dense labels. Reader state is
lazy and dropped on pickling; spawned workers open independent readers. Call
`dataset.close()` when finished. The loader factory explicitly uses `spawn` for
worker counts >0 and seeds Python, NumPy and PyTorch worker RNGs.

## Leakage and overlap boundaries

Pass **only `batch['X']`** into a temporal model, or only `flat['X']` into a
non-temporal model. Do not pass/stringify the full batch, metadata, original
filenames, class vocabulary, masks, annotations, tasks, fingerprints or targets.
Generic `TimeFTorchDataset` items expose tasks and annotations and are not used.
The connector whitelist rejects label-bearing Signals, even if the Signal count
remains 24. Sentinel tests modify target IDs, vocabulary text and fingerprint
metadata in memory and verify identical feature tensors; source files are never
changed. Auxiliary segments/launch fields/fingerprints are not returned by this
baseline adapter; they remain intact in the validated TimeF artifact.

Primary recognition and overlap remain separate objectives. Initial L1 baseline
training/evaluation uses **only `valid_l1` bins**; -1 is never a negative class.
A default PyTorch cross-entropy loss does not use this dataset's -1 convention;
a future trainer must use explicit masking or `ignore_index=-1`.

`overlap_l1_valid` is an explicit **observed-membership coverage** policy:
`workload_l1_multihot.any(-1)`. It is not a new upstream validity flag or proof of
exhaustive negative supervision. Zero overlap within that mask means at most one
**annotated** L1 class, not proven absence of concurrency. All-zero multi-hot bins
are not silently trained as negatives. Future overlap experiments must review
negative supervision and state coverage separately. Same-class concurrent kernels
may not yield multi-class overlap. Device-marker envelopes are not proof of
instruction-level simultaneity, and counter evidence does not establish causation.

## Normalization and Phase 1 compatibility

The existing `outputs/phase1/task7/train_statistics.json` is reused verbatim.
The verification command checks its SHA256 against the saved Task 7 report.
`NormalizationStats.validate_sources` checks the official training split hash,
exact training-input hash inventory and channel order. No fitting is implicit.
The command independently recomputes the same moments from **TimeF training
counters only** and compares them to the saved values with rtol/atol 1e-12.
It does not save or overwrite statistics or aggregate validation statistics.

Definition: per-channel population z-score over all 1,124 × 512 = 575,488 training
bins, including unlabeled and inactive bins; float64 parallel variance accumulation,
ddof=0; subtract mean/divide scale in float64, then cast to float32. Standard
deviations <=1e-12 use scale 1. No label/active-mask selection, timestamp-duration
weighting, clipping or per-trace display normalization is used. Raw float32
measurements remain separately available. Validation uses exactly the same
training statistics. Counters retain their upstream physical scaling before this
optional z-score.

Differences from Phase 1 are storage/API only: batched TimeF reads replace NPZ
input reads; `X`/`raw_X` replace `features`/`raw_features`; minimal bookkeeping
replaces variable-length auxiliary metadata; observed overlap coverage is made
explicit; default normalization reuses verified training statistics instead of
being disabled. Primary/multi-hot values, masks, raw counters and exact ns
alignment must match the original loader/source. The old loader is unchanged.

## Reproduce and use

From the Kernel X-ray root, the two views have already been prepared. To build
on a fresh workspace **only when their output paths do not exist**, run:

```bash
.worktrees/timenet-kernelsight/.venv/bin/uv run \
  --project .worktrees/timenet-kernelsight \
  python scripts/prepare_phase3_timef_views.py --split train
.worktrees/timenet-kernelsight/.venv/bin/uv run \
  --project .worktrees/timenet-kernelsight \
  python scripts/prepare_phase3_timef_views.py --split val
```

Builders refuse existing output directories, guard NPZ paths to the selected
partition and retain a pending marker until independent integrity passes. They
use the previously approved conversion safety limits of 3 GiB RSS and 2 GiB
minimum free disk. They do not access the mixed original counter files. Run
verification (without rebuilding the views):

```bash
.worktrees/timenet-kernelsight/.venv/bin/uv run \
  --project .worktrees/timenet-kernelsight \
  python scripts/verify_phase3_data_pipeline.py
```

No activation, Homebrew, pip install, model training or original-artifact rebuild is needed.
The command runs new tests, checks all 1,124 training and 160 validation records
against source NPZs, checks the old statistics using training only, benchmarks
CPU loaders in fresh processes and verifies preservation. Tests guard actual
TimeF Record decoding and NPZ reads against the real test partition. Checksumming
all accepted artifact files reads bytes without decoding test arrays. The spawned
worker smoke test needs PyTorch shared-memory/IPC permission; restrictive app
sandboxes can block `torch_shm_manager`. The default `num_workers=0` loader works
without that worker IPC. Optional CUDA/MPS tests run only when the backend is
available in the executing process.

For an interactive script in the TimeNet environment, add this repository's
`src/` to `PYTHONPATH`, then:

```python
from kernel_xray.phase3 import make_dataloaders, flatten_labeled_bins

loaders = make_dataloaders(batch_size=16, seed=7)
try:
    batch = next(iter(loaders['train']))
    temporal_X = batch['X']       # [16,24,512], full original temporal order
    bins = flatten_labeled_bins(batch, 'l1')
    flat_X, y = bins['X'], bins['y']  # [N,24] and [N], with all -1 removed
finally:
    for loader in loaders.values():
        loader.dataset.close()
```

Repeated constructions with the same seed produce the same epoch sequences;
training order changes deterministically between epochs as its generator advances.
Validation is never shuffled. Flattening happens **after** selecting trace
partitions; never randomly split the flattened bin pool into train/val/test.

## Validation results and next task

Exact measurements and the training-only L1 distribution are stored in
`outputs/phase3/task1/data_pipeline_validation.json` (local, not published).
The final strict-view verification checked **all 1,124 training and 160 validation
traces** against source counters, labels, masks, vocabularies and exact timestamp
arrays. Training-only mean/std recomputation matches Phase 1 exactly (maximum
absolute differences both 0); the original statistics SHA256 remains
`66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`.

| Check / measurement | Final result |
| --- | --- |
| New pipeline tests | 16 passed, 1 skipped (CUDA unavailable) |
| MPS tensor transfer | Passed |
| Original Phase 1 tests | 32 passed |
| Existing visualization verification | Passed; no plots regenerated |
| Protected earlier workspace files | 22,051 unchanged |
| Original Phase 2 TimeNet implementation | 520 files unchanged |
| CPU train epoch, batch 16 / 0 workers | 1,124 traces / 71 batches, 16.21 s, 69.35 traces/s |
| CPU validation epoch | 160 traces / 10 batches, 2.29 s, 69.86 traces/s |
| Fresh CPU loader process peak RSS | 1,712,160,768 bytes (1.595 GiB) |
| Spawned-worker compatibility | Two 16-trace batches in each partition passed |
| Worker smoke parent / largest-child peak | 1.442 / 0.423 GiB, separate observations |
| Train / val compiled view sizes | 394,066,578 / 60,049,749 bytes |
| Train / val view builder peak | 2.009 / 0.590 GiB, below the 3 GiB conversion stop |

Throughput includes decoding, normalization and collation but excludes index
construction. Peak RSS includes startup in a fresh process. Worker-smoke timings
include spawn/shutdown and are not full-epoch throughput estimates. Child peaks
are not summed or presented as a job-tree RSS measurement. The default remains
`num_workers=0`. RSS profiling uses macOS APIs; data tensors do not require macOS.
The initial full-artifact unbounded-index diagnostic peaked at 3.178 GiB;
index bounding and strict partition storage reduce the final observation above.
These are individual observations in different execution contexts, not a
controlled performance guarantee. No trace/bin sampling or supervision reduction
was used. Sandbox restrictions blocked initial worker IPC and masked MPS
availability; final checks ran with the required system permissions.

The separate views contain no test sources or counter rows. Their
`VIEW_VERIFIED.json` markers certify completed source comparisons; `view.json`
retains build hashes/provenance. `storage_isolation.json` records the original
mixed group and final pure groups using metadata only. The old full artifact
and all prior Phase 1/2 outputs remain byte-for-byte unchanged.

Training-only primary L1 support retains all vocabulary IDs:

| ID | Class | Valid training bins |
| --- | --- | ---: |
| 0 | matmul | 122,535 |
| 1 | conv | 8,535 |
| 2 | activation | 15,738 |
| 3 | normalization | 9,477 |
| 4 | softmax | 2,875 |
| 5 | pooling | 10,680 |
| 6 | reduction | 21,123 |
| 7 | attention | 7,046 |
| 8 | loss | 5,398 |
| 9 | elementwise | 11,106 |
| 10 | memory_movement | 19,308 |
| 11 | other | 0 |

There are **233,821 valid L1 bins** and **341,667 unknown bins**: 40.6300%
training coverage. Unknown remains -1. Normalization still uses all 575,488
training bins. Validation class distributions do not fit preprocessing or drive
API choices.

Phase 3 Task 2 should train the first **seeded PyTorch 24→12 per-bin linear
softmax baseline (multinomial logistic regression)** on normalized, valid-primary
L1 training bins. Evaluate validation macro-F1, per-class precision/recall/F1,
support, aggregate accuracy and labeled-bin coverage, with all 12 vocabulary IDs
kept explicit. Do not split bins across partitions, select a test-driven policy,
add labels as features or interpret this primary classifier as an overlap model.
Test remains reserved for a separately authorized final evaluation. Model fitting
and any optimizer/epoch choices belong to Task 2; none is performed here.
