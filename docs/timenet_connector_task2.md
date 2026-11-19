# Phase 2 Task 2 — local KernelSight connector

Implementation is in the separate TimeNet worktree
`.worktrees/timenet-kernelsight`, branch `feature/kernelsight-v4`, based on
`9bd1613019ee61fbc52e02106a1a45990ead7a54`. The original `~/Desktop/TimeNet`
checkout remains on its original branch without source changes. No commit, push,
PR, publication, model training or real validation/test conversion was performed.
Namespace and task-profile acceptance by TimeNet remain pending.

The worktree uses uv 0.12.5 / Python 3.13.4, with its own environment. System uv
0.6.14 was too old for TimeNet's >=0.10.10 requirement; a compatible version was
bootstrapped in the project cache before syncing the worktree. Kernel X-ray's
Python 3.9.6 environment was not upgraded or installed into.

## What works

- Pinned local-source acquisition checks all three official split JSONs without
  opening held-out NPZs. A compressed source manifest retains the independently
  verified SHA-256 inventory of the whole source snapshot. Source revision remains
  `d8cd8e804e5b619aadbb082a1fb3a8791b60a680`.
- One Record/Source per selected trace and 24 named scalar float32 Parquet Signals;
  exact stored counter values, channel ordering and 512 samples per channel.
- A canonical encoder/inverse decoder preserves every field of the input, labels
  and fingerprint NPZs, including original dtype/shape, all 513 ns timestamp edges,
  -1 labels, masks, complete vocabularies/parents, segments, launch and auxiliary data.
- Microsecond native coordinates follow the reviewed floor rule, with exact ns
  retained separately and an explicit final recording boundary. No rescaling.
- Four native task families use the supported reiterable streaming interface:
  primary L1/L2 classification per valid fixed bin and sparse L1/L2 multi-hot
  localization per trace. Overlap is not reduced to one class. All native tasks
  retain the official owner split.
- A whitelist-only `model_features(record)` returns only the [24,512] counter
  matrix. Target metadata, annotations, filenames and class IDs are excluded.
  It explicitly uses full counter context; native scopes identify the supervised
  bin, and are not falsely claimed to crop TimeNet's torch items automatically.

The inverse decoder promises exact NPZ **array** reconstruction, not reproduction
of ZIP container compression. This was verified for all selected synthetic and
training arrays, not for unconverted held-out tensors.

## Measured training-only scale result

Full machine-readable measurements are in
`outputs/phase2/task2/training/report.json`. The registry is retained alongside
that report. These are local partial research artifacts under a provisional
1.0.0 card, not complete or publishable KernelSight releases.

| Measurement | Result |
|---|---:|
| Training Records | 1,124 |
| Counter Signals | 26,976 |
| Primary L1 tasks | 233,821 |
| Primary L2 tasks | 233,821 |
| Sparse L1 localization tasks | 1,124 |
| Sparse L2 localization tasks | 1,124 |
| Total tasks | 469,890 |
| Record conversion | 29.03 s |
| Task-factory iteration | 2.54 s |
| TimeF write | 55.79 s |
| Native reader task iteration/checks | 17.62 s (~26,670 tasks/s) |
| Complete reader/source verification | 40.32 s |
| First Record eager read + feature extraction | 35.5 ms |
| Four-Record eager read + feature extraction | 48.8 ms |
| Compiled logical file size | 395,377,222 bytes (~377.1 MiB) |
| Full-process peak RSS, macOS bytes | 4,071,309,312 bytes (~3.79 GiB) |
| Source files hash-checked unchanged | 3,372 training NPZs |

These are individual observations, not statistically estimated production latencies.
The batch read followed a single read and may benefit from warm caches. The native
reader audit reuses one Record at a time, bounds task batches, and verifies every
class target and positive-run interval, not merely counts. Measurements were on
this arm64 macOS host using Python 3.13.4 / NumPy 2.5.0. Arrow reported denied CPU
capability sysctls in the restricted benchmark process; timings apply to that
execution environment. No speedup or memory extrapolation for the unconverted
held-out partitions is claimed.

### Resource review gate

**Correctness passed; the provisional memory criterion failed.** Peak RSS exceeds
2 GiB. Compiled size is below 1 GiB. A report status of `passed` denotes integrity
checks; its explicit `thresholds` block flags memory review as required.

A separate fresh-process diagnostic (`outputs/reports/task2_memory_diagnostic.json`)
measured 1,441,759,232 bytes peak after conversion and 1,441,792,000 after the
entire task-factory pass (~1.34 GiB). Thus materialized canonical metadata already
has a meaningful cost; task construction itself does not explain the full 3.79 GiB
peak. The complete writer/reader audit raised the peak further. This is diagnostic
evidence, not allocation-level proof that any particular TimeNet component is at
fault.

Recommended next investigation, preserving the task contract:

1. Measure RSS separately during metadata encoding, writer/catalog batching,
   DuckDB insertion and reader verification. Try bounded writer batches and spill
   settings only through supported or separately reviewed upstream changes.
2. Profile Python-list canonical payload retention and transient JSON copies;
   consider a reviewed typed-payload storage design if metadata dominates. This
   changes the canonical storage schema and needs explicit version/review.
3. If necessary, compare a separately versioned compact task profile using known
   homogeneous intervals. That changes evaluation units and exposes ground-truth
   boundaries, so it cannot silently replace fixed-bin classification.

No task family was removed, no bins were sampled away, and no resource threshold
was quietly relaxed. A controlled full-dataset integrity conversion is
**functionally supported but resource-gated**: do not treat this as meeting a 2 GiB
budget. Resolve optimization or explicitly approve a larger measured budget first.
Publication remains separately blocked on namespace and licensing requirements.

## Reproduce stages A–C

From the TimeNet worktree, use its environment explicitly:

```bash
cd "/Users/nawaf/Desktop/Kernel X-ray/.worktrees/timenet-kernelsight"
export PATH="$PWD/.venv/bin:$PATH"
export UV_CACHE_DIR="/Users/nawaf/Desktop/Kernel X-ray/.cache/uv"
export KERNELSIGHT_ROOT="/Users/nawaf/Desktop/Kernel X-ray/data/kernelsight_v4"
export KERNELSIGHT_PARTITIONS=train
```

Stage A, independently synthetic fixtures only (the two real tests skip unless
explicitly enabled):

```bash
unset KERNELSIGHT_TEST_ROOT
uv run pytest packages/timenet-connectors/src/timenet_connectors/\
datasets/williamhtan/kernelsight/tests -q
```

Stage B, opt-in real training tests and persistent prototype registry:

```bash
export KERNELSIGHT_TEST_ROOT="$KERNELSIGHT_ROOT"
uv run pytest packages/timenet-connectors/src/timenet_connectors/\
datasets/williamhtan/kernelsight/tests -q
uv run python benchmarks/kernelsight/verify_training.py \
  --data-root "$KERNELSIGHT_ROOT" --stage prototype \
  --output "/Users/nawaf/Desktop/Kernel X-ray/outputs/phase2/task2/prototype-new"
```

Stage C, only after Stage B passes:

```bash
uv run python benchmarks/kernelsight/verify_training.py \
  --data-root "$KERNELSIGHT_ROOT" --stage training \
  --output "/Users/nawaf/Desktop/Kernel X-ray/outputs/phase2/task2/training-new"
```

Output directories must be fresh; scripts refuse to overwrite registries. Defaults
and benchmark entry points use train only. All official JSON indexes are read for
split isolation; held-out NPZs are never decoded. A read guard enforces the selected
training file set. The callable connector supports explicit additional partitions
for a later controlled run, but those real paths have not been converted here.

## Automated acceptance and contributor checks

See `outputs/reports/task2_acceptance.json` for final check status and the matching
logs. Connector tests cover discovery/source pinning, signed-zero bit fidelity,
canonical shape/dtype reconstruction, exact ns/floor arithmetic, overlap and empty
outputs, all synthetic split identities, class/mask corruption, path/hash failures,
input leakage sentinels, deterministic task streams, and actual reader round trips.
The scale audit independently reconstructs all training NPZ arrays and all native
supervision from stored tasks. Phase 1 regressions run in the original environment.

The default `make license-check` exited successfully but returned an empty table,
so it is **not** treated as a substantive license scan. Investigation found the
audit tool's alternate-Python discovery splits sys.path on spaces; this project
path contains spaces. An in-process invocation in the actual worktree interpreter
produced a nonempty inventory and passed the same denied-license filter. Both
outputs are retained. Dependency scanning does not resolve dataset notice rights.

Pre-commit `--all-files` sees tracked files; new connector files are still untracked
because no commit/staging was requested. Hooks were therefore additionally invoked
with the explicit new-file list. Formatting/type checks also include untracked
source. The 611 KiB hash inventory was gzip-compressed deterministically to about
178 KiB to stay below the large-file hook limit; it contains hashes, not dataset
bytes. Compression does not change the decoded hash inventory.

The initial sandboxed core test attempt stalled starting the local S3 mock server
and was interrupted. The full suite was retried with localhost permissions; report
that retry separately rather than count the interrupted run as a pass. No TimeNet
core sources were changed to bypass tests.

## Files added or modified

In the TimeNet worktree, all implementation additions are uncommitted:

- `datasets/williamhtan/__init__.py` and `datasets/williamhtan/kernelsight/`
  beneath `packages/timenet-connectors/src/timenet_connectors/`.
- Connector `__init__.py`, `connector.py`, `dataset.yaml`, `requirements.txt`, README.
- `source.py`, compressed `source_manifest.json.gz`, `mapping.py`, and `tasks.py`.
- Tests `__init__.py`, `conftest.py`, `test_connector.py`, and `test_real.py`.
- `benchmarks/kernelsight/verify_training.py`.

In Kernel X-ray: `.gitignore` excludes the worktree; AGENTS.md and README link this
implementation/status; this document records commands and limitations. Generated
registries/reports/environment inventories are under `outputs/phase2/task2/` and
`outputs/reports/task2_*`. Original source, splits, upstream tools, completed Phase 1
outputs and original environment are preserved. No raw dataset bytes are checked
into the connector fixtures or copied into TimeNet source.
