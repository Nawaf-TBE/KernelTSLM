# Phase 2 Task 2.1 — memory profiling and optimization

## Outcome and scope

**Memory target achieved; ready for controlled full conversion.** This conclusion
applies to the measured **training-only benchmark**. Full 1,444-trace conversion
has not been performed or demonstrated to fit 2 GiB. The final uninstrumented
combined build/read audit peaked at **2,097,987,584 bytes (1.954 GiB)**; a separate
instrumented run of the final implementation peaked at **2,100,477,952 bytes
(1.956 GiB)**. The latter leaves only **44.83 MiB** below the provisional 2 GiB
threshold. This is a measured improvement, not a portable memory guarantee.

All 1,124 Records, 26,976 counter Signals, 3,372 training source NPZ mappings and
469,890 native tasks still pass exact round-trip checks. No real validation/test
NPZ was decoded, no full conversion or training was run, and nothing was committed,
published, pushed or submitted as a PR. Official split identities remain unchanged.
Namespace/publication and source-notice questions from the decision review remain
open; this memory result does not resolve those questions.

## Evidence and reproducibility

The working branch is `feature/kernelsight-v4` in
`.worktrees/timenet-kernelsight`, based on TimeNet commit
`9bd1613019ee61fbc52e02106a1a45990ead7a54`. The original sibling TimeNet checkout
remains clean. Source data is pinned to
`williamhtan/kernelsight@d8cd8e804e5b619aadbb082a1fb3a8791b60a680`.
Measurements used the existing separate TimeNet Python 3.13.4 environment,
NumPy 2.5.0, DuckDB 1.5.6 and Arrow 24.0.0 (mimalloc). The Phase 1 Python 3.9
environment was preserved.

All evidence lives under `outputs/phase2/task2_1/` (local, not published):

- `outputs/phase2/task2_1/comparison.json` (local, not published): original result,
  experiments, counts, timings, sizes and throughput.
- `baseline_profile/memory.json` (`outputs/phase2/task2_1/baseline_profile/memory.json`, local, not published)
  and `text_reuse/memory.json` (`outputs/phase2/task2_1/text_reuse/memory.json`, local, not published):
  stage events, high-water RSS, sampled RSS, Arrow allocations, call counts,
  elapsed times and retained-object observations.
- `final_unprofiled/report.json` (`outputs/phase2/task2_1/final_unprofiled/report.json`, local, not published):
  final complete correctness/resource benchmark, without profiling hooks.
- `python_allocations_32/` and `python_allocations_representative32/`: limited
  Python-allocation diagnostics, excluded from budget claims.
- `baseline_code/`: original connector, benchmark and the two original core files;
  original Task 2 compiled artifacts also remain untouched.
- `preservation_before.json`, `preservation_after.json`, `code_inventory.json`,
  `core_changes.patch` and `acceptance.json`: preservation and final provenance.

`benchmarks/kernelsight/profile_memory.py` wraps existing calls temporarily,
without retaining their arguments/results. On macOS it measures the current
process through `proc_pidinfo`, samples RSS every 20 ms, records `ru_maxrss`
(bytes on macOS), and records `pyarrow.total_allocated_bytes()`. High-water RSS
captures peaks missed between samples. Measurements are **per process**, not a
sum of unrelated concurrent test processes. Elapsed call durations are inclusive;
nested call times and maximum stage RSS must not be added together. Canonical validation also occurs during the final audit, so its aggregate
call maxima can occur after conversion. Raw NPZ loading calls shown below measure
conversion; reader-side NPZ parsing is included in the complete verification time.

Fresh processes were used for each full experiment and for independent reader
checks. Some diagnostic runs overlapped independent tests; these timings are
observations, not statistically controlled performance estimates. Limited
`tracemalloc` runs change allocation behavior and are not used to claim a 2 GiB
result. The first used 32 initial traces; the later one used 32 evenly spaced
training entries. Current `--limit` reproduces the latter selection.

## Stage measurements and causes

The following figures are from the final **instrumented** run. RSS and high-water
are MiB. Each stage's machine-readable events and aggregate calls provide the
exact byte counts; short initialization calls may have no 20 ms sample.

| Stage | Elapsed measurement | RSS observation | High-water / retention observation |
| --- | ---: | ---: | --- |
| Raw NPZ loading/parsing | 2.41 s summed over 1,124 calls | maximum single-call increase 2.05 MiB | incremental calls while accumulated Records already exist; not a stand-alone raw heap total |
| Canonical array construction | 1.58 s over 38,216 calls | maximum single-call increase 2.77 MiB | full conversion ends at 1,083.3 MiB |
| Record construction | 11.27 s over 1,124 calls | maximum single-call increase 4.39 MiB | includes required JSON validation/copy; all 1,124 Records retained during writing |
| Signal creation | 0.27 s over 26,976 calls | maximum single-call increase 0.016 MiB | raw float32 values remain present |
| Native task-factory traversal | 2.59 s | 1,108.6 → 1,109.0 | no material growth from collecting all tasks; tasks are streamed |
| TimeFWriter initialization | 0.00020 s | 1,109.2 → 1,109.2 | no significant allocation at initialization |
| Counter Parquet writing | 1.47 s | 1,111.8 → 1,231.6 | high-water 1,231.6 |
| DuckDB hierarchy/control serialization | 55.34 s inclusive | 1,231.6 → 1,930.3 | process high-water 2,003.2; includes tasks and finalization |
| Canonical JSON serialization | 12.84 s summed, nested above | maximum individual increase 2.14 MiB | 47 bounded Record inserts; Arrow pool does not accumulate |
| Native task serialization | 39.39 s, nested above | 1,823.0 → 1,746.6 | high-water 2,003.2 during this stage; Arrow pool zero at boundaries |
| TimeFReader initialization | 0.000044 s | 1,706.1 → 1,706.1 | inherited build allocator RSS, not reader initialization allocation |
| Complete stored-task/source verification | 41.81 s inclusive | 1,706.1 → 1,056.2 | dataset and all converted Records already dead by weak reference |

The unoptimized fresh baseline finished conversion at **1,339.4 MiB**. Its sole
Record-table flush grew RSS from **1,585.2 to 3,272.2 MiB** (1,687 MiB increase).
Before task serialization, the Arrow allocator held **368.0 MiB**; after hierarchy
serialization it held **521.5 MiB**, released at connection close. Merely streaming
native tasks did not prevent those allocations.

Two independent problems were measured:

1. The 10,000-row hierarchy buffer accumulated all 1,124 very large canonical JSON
   rows at once. Python JSON strings, Arrow conversion, and DuckDB insertion then
   overlapped. An 8 MiB payload cap reduced the largest Record-flush RSS increase
   to about 68 MiB and changed one large insert into 47 inserts.
2. `register` / `INSERT` / `unregister` inside the hierarchy transaction retained
   Arrow scan inputs in transactional catalog state. Even bounded batches kept
   accumulating until commit. A reproducer accumulated roughly 1 MiB of Arrow
   pool allocation per insert; the regression test failed on the old code after
   the second insert. Using DuckDB's public relation API released each scan input
   after insertion, with the same enclosing transaction and rollback semantics.

Python allocations also matter. The two 32-trace diagnostics place required
Pydantic JSON-container allocation at about 15–19 MB, with NumPy-decoded text and
Python lists also prominent. These small samples identify sources, not an exact
partition or extrapolation of full-process memory. `tracemalloc` observes
Python-visible allocations (including some extension-managed buffers); Arrow's
pool observes its allocator only. Neither includes every DuckDB/native allocator,
thread stack, library mapping or arena. Subtracting either from RSS would not give
an exact native/Python split.

Weak references show the converted Dataset and all 1,124 Records are gone before
reader verification, while RSS remains high. Explicit collection does not return
all native/allocator pages to the OS. Independent fresh readers peak at
**0.454–0.840 GiB**, including complete verification; the original Task 2 registry
verifies at 0.524 GiB. Therefore the combined reader RSS largely inherits build
allocation history. The reader alone is not the cause of the 3.79 GiB peak.

## Changes retained

### Connector allocation reductions

In `packages/timenet-connectors/src/timenet_connectors/datasets/williamhtan/kernelsight/mapping.py`:

- Validate array dtype/content directly instead of calling an encoder whose
  discarded `.tolist()` output duplicated every value, including counter arrays.
- Remove the explicit JSON validation pass immediately before `Record(...)`.
  Record's existing constructor still performs the same required validation and
  container copy. No validation or schema invariant was disabled.
- Reuse equal immutable strings in fresh canonical lists through a bounded
  4,096-entry LRU cache. Lists/dictionaries remain independent, including across
  Records; numeric values, dtype/shape descriptors and serialized text are
  unchanged. This is an entry cap, not a universal byte cap for arbitrary text.

Removing duplicate validation improved conversion time but did **not** by itself
reliably reduce peak RSS. Text reuse reduced measured end-of-conversion RSS by
about 256 MiB versus the fresh baseline. The irreducible current architecture
still retains all converted Records, canonical Python containers and Signals
through writing. No lazy reconstruction scheme or external sidecar format was
introduced to evade that cost.

### Minimal core changes justified by the profile

Only two TimeNet core files changed:

- `packages/timenet/src/timenet/format/control_writer.py`: `_TableBatch` now flushes
  at either the existing 10,000-row cap or an estimated 8 MiB encoded payload.
  UTF-8 strings/binary payloads are measured; scalar estimates use eight bytes.
  This is a batch payload target, not an RSS bound. One oversized row remains
  indivisible. Failed inserts retain their diagnostic buffer. Task batch coverage,
  transaction boundaries and output schema do not change.
- `packages/timenet/src/timenet/format/table_writer.py`: convert with the same
  Arrow schema, then `connection.from_arrow(batch).insert_into(table.name)`.
  Avoid transactional catalog DDL. Column order, error translation, empty inputs
  and rollback remain tested. No commit is inserted between batches.

Six independent regression tests cover byte/row caps, Unicode sizing, oversized
rows, failed insertion, Arrow lifetime across 16 inserts, column order and rollback.
`retention_regression_before.txt` records failure against the original code.
The full core suite passed after both fixes. These are allocation/lifetime changes,
not TimeF format/API or scientific-supervision changes.

### Alternatives investigated

Releasing temporaries/collecting garbage does not cure transaction-retained Arrow
inputs or resident allocator pages. The existing streaming task factory is already
appropriate; bounded hierarchy buffers address the distinct serialization problem.
Fresh-process write/read modes are available for operational isolation, with
write-only reports explicitly **not** claiming complete reader verification.
Each write-only experiment below has an independent complete reader check.

TimeFWriter publicly exposes `row_group_target_bytes`. The default is **4 MiB**.
We tried 1 MiB and 8 MiB without changing production defaults. The directory named
`small_rowgroups_write` actually used **8 MiB, larger than default**; its historical
name is retained but the recorded configuration is corrected. Results were not
reliable enough to prefer either option over the default.

DuckDB's measured defaults are 19.1 GiB `memory_limit`, 12 threads and
`preserve_insertion_order=true` (see `native_settings.json`). The existing writer
has no public DuckDB memory-limit/thread configuration argument. A direct DuckDB
`SET` is supported by DuckDB but not exposed on this writer; no hidden connection
patch, allocator purge or broad configuration API was added. A DuckDB buffer limit
would not cap canonical Python or Arrow memory anyway. This task fixes the
specific measured retention issue instead of introducing speculative settings.

## Experiment comparison

GiB/MiB use powers of 1024. Combined rows include full reader verification. All
use the same four native task families and 1,124 training traces. Row-group trials
precede immutable-text reuse. Complete exact reports, not rounded values, are in
`comparison.json`.

| Experiment | Peak GiB | Convert s | Write s | Compiled MiB | Reader tasks/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Task 2 | 3.792 | 29.03 | 55.79 | 377.06 | 26,670 |
| Fresh unoptimized profile | 3.305 | 30.36 | 57.36 | 375.81 | 26,318 |
| Remove duplicate validation/list copies | 3.758 | 18.40 | 56.54 | 375.81 | 26,626 |
| Add byte-bounded hierarchy inserts | 2.564 | 19.65 | 57.81 | 375.81 | 25,549 |
| Add relation insertion | 1.987 | 18.83 | 58.34 | 375.81 | 26,366 |
| Repeat preceding configuration, unprofiled | 2.005 | 18.90 | 60.35 | 375.81 | 25,195 |
| Add immutable-text reuse, profiled | 1.956 | 19.40 | 57.28 | 375.81 | 25,002 |
| Final combined, unprofiled | **1.954** | **19.02** | **57.19** | **375.81** | **26,051** |
| Fresh reader of original registry | 0.524 | — | — | 377.06 | 26,659 |
| 8 MiB row groups, write-only | 2.138 | 19.12 | 58.10 | 375.70 | — |
| Independent reader of 8 MiB build | 0.454 | — | — | 375.70 | 25,168 |
| 1 MiB row groups, write-only | 1.989 | 20.40 | 60.05 | 375.84 | — |
| Independent reader of 1 MiB build | 0.655 | — | — | 375.84 | 25,060 |
| Independent final reader with explicit trace-coverage guard | 0.840 | — | — | 375.81 | 27,634 |

Final peak falls about **48.5%** versus historical Task 2 and **40.9%** versus the
fresh profiled baseline. Conversion is about 35% faster than historical Task 2;
writing/reader throughput remain similar, with no speedup claim beyond these
observations. Physical compiled size differs slightly by serialization/batching
and row-group configuration; raw reconstruction, not compiled byte identity, is
the correctness criterion. Neither historical nor fresh baseline is discarded.
The pre-text-cache repeat crossing 2 GiB demonstrates why one narrow pass should
not be treated as a robust budget guarantee.

## Correctness, preservation and tests

Final complete audit (`final_unprofiled/report.json`):

- 1,124/1,124 training traces, 26,976/26,976 Signals, all three source NPZ mappings
  per trace reconstructed in every key, dtype, shape and value; numeric bit
  patterns also checked. Absolute int64 nanosecond edges remain exact.
- 233,821 L1 and 233,821 L2 fixed-bin ClassificationTasks; 1,124 L1 and 1,124 L2
  localization tasks. Reader targets reconstruct primary arrays including -1,
  both multi-hot matrices, original vocabulary order, and native bin scopes.
- All tasks retain official TRAIN identity, correct owner, empty input annotations
  and absent rationales. Counter-only `model_features` whitelist and synthetic
  leakage tests remain enforced. Targets/segments/fingerprints stay supervision
  or metadata, never counter features. Native microsecond quantization remains
  explicit; exact original ns values are preserved separately, as in Task 2.
- Runtime NPZ allowlist permits only selected training paths. Official held-out
  split support remains covered synthetically. Hashing held-out files for
  preservation reads bytes without decoding arrays.
- SHA256 preservation check: **21,524 pre-existing files unchanged**, including
  all 4,341 raw dataset files, 17,099 Phase 1 environment files, 48 Phase 1/plot
  artifacts and 36 original Task 2 artifacts. No missing files; original TimeNet
  checkout has an empty `git status --short`.

Checks run:

| Check | Result |
| --- | --- |
| `make test` | 2,896 passed, 2 skipped, 5 warnings; 285.84 s |
| Final real-training connector + six core regression tests | 23 passed, no skips; 1.63 s |
| `make test-connectors` | All connector environments/types pass; KernelSight default mode 15 passed, 2 opt-in skips |
| `make lint-fix`, final `make check` | Passed |
| `make check-ci`, explicit hooks over new files | Passed, including Python 3.11 typing |
| Root Phase 1 pytest | 32 passed; 12.20 s |
| Existing visualization verification | Passed; galleries unchanged |

The final independent reader audit also explicitly rejects duplicate paths and
asserts that every selected source trace was visited. Its 0.840 GiB peak illustrates
fresh-process allocator variability while remaining below the combined build peak.

The full core suite ran after the core fixes; the final connector suite ran after
the additional immutable-text cache. Core was unchanged by that connector edit.
Two existing core skips and fork-related warnings are recorded in the full log.

## Reproduce without touching the existing registries

From the TimeNet development worktree, use its existing uv environment. Each build
must have a fresh output path; replace `reproduce-*` if already used. These commands
use only training and never perform a full conversion:

```bash
cd '/path/to/kernel-xray/.worktrees/timenet-kernelsight'
export UV_CACHE_DIR='/path/to/kernel-xray/.cache/uv'
export KERNELSIGHT_ROOT='/path/to/kernel-xray/data/kernelsight_v4'
export KERNELSIGHT_TEST_ROOT="$KERNELSIGHT_ROOT"
export KERNELSIGHT_PARTITIONS=train

.venv/bin/uv run python benchmarks/kernelsight/verify_training.py \
  --data-root "$KERNELSIGHT_ROOT" --stage training \
  --output '/path/to/kernel-xray/outputs/phase2/task2_1/reproduce-combined'

.venv/bin/uv run python benchmarks/kernelsight/profile_memory.py \
  --data-root "$KERNELSIGHT_ROOT" \
  --output '/path/to/kernel-xray/outputs/phase2/task2_1/reproduce-profile'

.venv/bin/uv run pytest \
  packages/timenet-connectors/src/timenet_connectors/datasets/williamhtan/kernelsight/tests \
  packages/timenet/tests/format/test_arrow_transaction_retention.py \
  packages/timenet/tests/format/test_hierarchy_byte_batches.py -q
```

For process isolation, run the benchmark once with `--mode write`, then launch a
new invocation with `--mode read --version /absolute/path/to/the/written/1.0.0`,
both with `--stage training` and distinct fresh `--output` paths. The same modes
are available on the profiler. A read-only result alone does not measure build RSS.
For a limited Python diagnostic add `--limit 32 --trace-python` to the profiler;
never compare that inflated tracing run to the production budget.

To reproduce the unoptimized baseline, use a **separate disposable copy** of the
base TimeNet checkout, overlay the preserved paths from `baseline_code/`, use the
same uv lock/environment, and run its archived `verify_training.py --stage
training` into a new directory. Do not overwrite this optimized worktree, the
original sibling checkout, or old benchmark registries. `core_changes.patch` and
`code_inventory.json` identify the final code alongside the archived baseline.

## Recommendation for the next authorized task

Proceed with a **controlled** full conversion only in a subsequent explicitly
requested task. Prefer a fresh writer process followed by a separate reader audit;
monitor actual writer RSS and disk space, keep source/split guards, retain the
partial artifact on interruption for diagnosis, and compare all four task-family
counts to independently derived expectations. Define a stop threshold before that
run. If 2 GiB is a hard limit, stop when reached rather than claiming the training
measurement proves full-dataset compliance. The additional 320 Records can raise
retained conversion memory; this task has not measured that increase.

No reduction of labels, bins, records, vocabularies, native task families, or dense
fixed-bin classification is warranted. Larger-scale streaming of Record creation
would be a separate architectural change if a strict full-dataset cap still
fails. The current measured training target is met without that redesign.
