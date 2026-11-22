# Phase 1 Task 7: model-ready records

`kernel_xray.dataset.KernelSightDataset` is a lazy, read-only PyTorch Dataset for
KernelSight v4. It does not instantiate OpenTSLM, train a model, or claim to be
an OpenTSLM-specific adapter. It follows the delivered v4 NPZ schema and official
split indexes. The upstream SASS stub is preserved; it expects an additional
`sass/sass_modality.npz` that is not present in this dataset distribution, so
this loader does not depend on it or reconstruct SASS features.

## Setup and reproducible verification

Use the existing Python 3.9.6 environment from the project root. Dependencies are
already installed here. For reproducing the setup in an existing project venv:

```bash
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python scripts/verify_model_records.py
```

The verification command runs the automated suite, fits training-only statistics,
checks raw and normalized DataLoader batches, and writes:

- `outputs/phase1/task7/train_statistics.json`: reusable statistics with source hashes.
- `reports/task7_tests.xml`: machine-readable pytest results.
- `reports/task7_verification.json`: versions, test counts, batch shapes, and integrity results.

The command hashes raw files, split files, upstream tools, and existing output
files before/after execution to verify preservation. It writes only Task 7
outputs and reports. Repeating it intentionally refreshes those Task 7 artifacts.
Normalization statistics JSON is deterministic for a fixed dataset/environment;
test timings in the XML can vary. The existing visualization dependency lock and
completed visualization outputs remain unchanged.

Run only tests or the training-only batch example:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/example_dataloader.py
.venv/bin/python scripts/example_dataloader.py --fit-normalization outputs/phase1/task7/train_statistics.json
.venv/bin/python scripts/example_dataloader.py --stats outputs/phase1/task7/train_statistics.json
```

The example uses a seeded PyTorch generator (seed 7), batch size 4, `shuffle=True`,
and `num_workers=0`. No transformation or random operation occurs in the Dataset.
Optional `--batch-size`, `--seed`, and `--num-workers` flags configure the example.
The verified configuration uses zero workers for portability on macOS. Worker
processes require the usual `if __name__ == '__main__'` guard; records are loaded
with context managers and no file handles are retained in the Dataset.

## Python API

```python
from kernel_xray import (
    KernelSightDataset, NormalizationStats, collate_records, fit_training_statistics,
)
from torch.utils.data import DataLoader
import torch

raw_train = KernelSightDataset(root='.', split='train')
# Fitting explicitly reads train inputs only; it accepts no alternate fit split.
stats = fit_training_statistics(root='.')
train = KernelSightDataset(root='.', split='train', normalization=stats)
loader = DataLoader(
    train, batch_size=4, shuffle=True,
    generator=torch.Generator().manual_seed(7),
    num_workers=0, collate_fn=collate_records,
)
batch = next(iter(loader))
x = batch['features']             # [4, 24, 512]; only this enters the model
raw_x = batch['raw_features']      # unchanged dataset values
l1 = batch['targets']['workload_l1']
valid_l1 = batch['masks']['valid_l1']
# No training or loss computation occurs in this example.
```

The loader accepts `split='train'`, `'val'`, or `'test'`, or the corresponding
**official JSON path**, e.g. `split='splits/train.json'` from the project root.
All three indexes are checked for counts, duplicate paths, and shared paths.
Analysis tags (`iid`, `param_ood`, `composed`, `length_ood`) and custom split files
are rejected. Record order follows the official JSON order. Construction is lazy:
no trace tensors are loaded until indexing. With normalization, construction
also checks all training source hashes without opening held-out NPZ arrays.

For later model selection and final evaluation, respectively, the same fixed
training statistics can be applied with:

```python
val = KernelSightDataset(root='.', split='val', normalization=stats)
test = KernelSightDataset(root='.', split='test', normalization=stats)
```

These lines are API documentation, not permission to inspect real held-out traces
for development. This task's real-data tests and examples use training only.
Synthetic fixtures exercise validation/test indexing and normalization, including
large held-out value changes that must not affect training statistics.

## Record and batch contract

An individual record is a dictionary with these fields. Shapes below are per
record; DataLoader prefixes every tensor shape with batch dimension `B`.

| Field | Shape | PyTorch dtype / meaning |
|---|---|---|
| `features` | `[24,512]` | float32; raw values or optional normalized copy |
| `raw_features` | `[24,512]` | float32; exact stored input counter matrix |
| `time_edges_ns` | `[513]` | int64; exact absolute timestamp edges |
| `targets.workload_l1` | `[512]` | int64; IDs 0–11 or -1 |
| `targets.workload_l2` | `[512]` | int64; IDs 0–72 or -1 |
| `targets.workload_l1_multihot` | `[512,12]` | float32; original binary L1 membership |
| `targets.workload_l2_multihot` | `[512,73]` | float32; original binary L2 membership |
| `targets.multihot_n_active` | `[512]` | int64; number of active L1 classes |
| `targets.overlap_l1` | `[512]` | bool; at least two active L1 classes |
| `targets.multihot_has_overlap` | scalar | bool; any multi-class bin in trace |
| `masks.valid_l1`, `masks.valid_l2` | `[512]` | bool; respective primary ID >= 0 |
| `masks.mask_labeled` | `[512]` | bool; exact original mask, agrees with valid L1 |
| `masks.mask_any_kernel` | `[512]` | bool; exact original kernel-activity mask |
| `masks.multihot_l1`, `masks.multihot_l2` | `[512]` | bool; any active label at that hierarchy |
| `metadata` | per-record dictionary | identity and variable-length annotations |

Metadata preserves channel names and their original order, L1/L2 vocabulary
order, L2-to-L1 parent mapping, split record/path, launch intervals, kernel names,
function indexes, segment arrays, and attribute/spatial vocabulary metadata where
present. `collate_records` stacks the fixed tensors but retains metadata as a
list of `B` dictionaries. It checks channel/vocabulary/parent order, split, and
normalization method before batching. Do not use default collation for these
variable-length launch/segment fields. Fingerprints are not loaded.

The model-facing field is **only `features`**. Targets, masks derived from labels,
filenames, kernel names, segment annotations, and other identity metadata must
not enter a counter-only model or explanation prompt. Keep these for supervision,
evaluation, provenance, and ground-truth inspection.

## Preprocessing and supervision decisions

- No reordering, cropping, padding, temporal resampling, smoothing, augmentation,
  clipping, or per-trace normalization. The matrix is channel-first `[24,512]`.
- Absolute int64 timestamps are preserved. The Dataset does not subtract a start
  time, convert to floating point, or infer a global sampling interval. Each bin
  corresponds to its original pair of adjacent edges.
- Input float32 measurements are copied exactly; primary int32 labels become
  PyTorch int64 for loss compatibility. Binary multi-hot values become float32;
  masks become bool. Unlabeled primary IDs remain **-1**, never a thirteenth L1
  or seventy-fourth L2 class. Mask them explicitly (or configure a future primary
  classification loss with `ignore_index=-1`). The default cross-entropy ignore
  value is not the dataset convention; no loss function is configured here.
- Primary targets never replace multi-hot targets. The loader checks that every
  valid primary label belongs to its multi-hot track, the L2 parents agree with
  L1, active counts match, and the overlap flag agrees with `>=2` active L1s.
- `mask_labeled` is not kernel activity. Multi-hot-positive bins may differ from
  primary-labeled bins. The `multihot_*` masks mark bins with known positive
  membership; they do not establish exhaustive negative supervision in unlabeled
  bins. Future overlap metrics must state their supervision/negative-bin policy.
- Same-class concurrent kernels may produce only one active L1 class, so this
  overlap target measures multi-class overlap, not every possible concurrency.
  Device-marker envelope labels do not establish instruction-level simultaneity.
- No label remapping or resampling is used to compensate for imbalance. Vocabulary
  entries absent from observed primary labels remain in their original positions.

## Optional normalization

The default is no additional normalization: stored counters already have the
upstream physical scaling. Optional z-score normalization fits a population mean
and standard deviation **per channel across all 1,124 × 512 training bins**.
Every bin has equal weight, including unlabeled and inactive bins. This is an
explicit label-independent choice that retains background/inactive measurements.
It does not weight bins by physical duration; traces have different time spans.

Fitting reads only training inputs, in official order, using float64 blockwise
mean/variance accumulation and `ddof=0`. It never loads labels or validation/test
inputs. Normalized features are `(raw - train_mean) / train_scale`, cast back to
float32. Channels with training std <= `1e-12` use scale 1, avoiding division by
zero while retaining novel nonzero held-out values. There is no clipping.
`raw_features` has separate storage and remains unchanged.

Serialized statistics include 24 channel names, means/stds, bin count, the
method/threshold, official training split SHA-256, and all training input paths
and SHA-256 hashes. Loading a normalized Dataset validates this training
provenance and checks channel order on each item. Never refit on held-out data.
Statistics JSON is a trusted local artifact; its hashes detect source changes,
not deliberate fabrication of fitted values. Refit with the provided command
when reproducing it. Statistics are separate from the visualization p99 display
scaling and do not overwrite raw files.

## Test coverage and evaluation boundary

The suite checks every real training record against source NPZ arrays for
features, timestamps, channel order, launch metadata, labels, masks, class ranges,
and overlap consistency. It independently compares real training mean/std to a
NumPy reference. Synthetic tests cover zero-variance handling, normalization
math, held-out isolation, stale-source rejection, corrupt-label rejection,
path/split rejection, serialization determinism, batch shapes, seeded shuffling,
pickling, storage independence, and channel-order mismatch rejection.
An autouse read guard fails any test that attempts to load a real held-out NPZ.
Val/test split isolation is checked from official JSON paths; their schemas are
exercised through synthetic records. This deliberately does not claim a full
real held-out tensor audit or model performance evaluation.
