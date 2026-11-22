# Kernel X-ray Phase 2 Task 1 — TimeNet integration plan

Research snapshot: 18 November 2026 (Asia/Muscat). This document is a design, not an
implemented connector. It describes the local `../TimeNet` checkout at commit
`9bd1613019ee61fbc52e02106a1a45990ead7a54`, whose `timenet` package declares version
0.12.0 and Python >=3.11. Git status was clean before inspection. No upstream
files, dataset files, partitions, dependencies, or models were changed for this task.

## 1. Findings and recommendation

Use one TimeF Record per original trace, containing one Source with **24 scalar
float32 Signals**, one per counter in original order. Preserve all 512 samples.
Use an explicitly documented microsecond timeline only as a derived convenience
view, and retain all **513 original int64 nanosecond edges** separately. Preserve
exact dense labels, masks, segment arrays, vocabularies, and auxiliary NPZ data
in a versioned, JSON-native canonical payload attached to the Record. Do not
make targets into extra input Signals.

Recommended first connector identity is `williamhtan/kernelsight` (subject to
namespace approval), compiled release `1.0.0`, upstream release `v4`, pinned to
an immutable Hugging Face source commit. The upstream v4 release number and the
TimeNet conversion release number are different version domains.

For the first implementation, recommend one **sparse TemporalLocalizationTask per
trace** for multi-hot L1 class intervals, carrying the original official split.
This gives every trace a task and exposes overlap without turning -1 into a
class. Keep the other targets losslessly in the canonical payload. L1/L2 primary
classification and L2 localization can subsequently be separate task families,
with an explicit version change. Task granularity needs agreement before Task 2;
the canonical conversion should not depend on that choice.

There are two material limitations: TimeF's native temporal coordinates cannot
represent nanosecond boundaries exactly, and it has no dedicated dense masked
sequence-labeling task type. The proposed payload avoids losing source
information, but consumers need a KernelSight-aware decoder to recover dense
supervision; generic TimeNet readers will not automatically interpret its schema.

## 2. Architecture and actual source APIs

TimeNet is a `uv` workspace with two packages. `timenet` supplies the TimeF model,
SDK, CLI, registries, engine, reader, writer, and BaseConnector.
`timenet-connectors` supplies recipes, dataset cards, discovery, download helpers,
and `timenet-build`. Building follows `metadata -> download -> convert ->
derive_schema -> store_dataset`; a connector returns objects and does not own
storage. The engine also supports `compose` for parent releases, which this base
conversion does not need.

A compiled release contains a DuckDB control plane, Parquet values by default
(or Zarr), packed payloads where applicable, and a manifest written as the commit
marker. Readers consume the manifest and stored objects without importing the
connector. Builds record package/interpreter versions and TimeNet code commits.
Releases in remote registries are immutable; local `--force` rebuilds do not grant
permission to replace published content.

Source references below are relative to this document in the sibling checkout;
all APIs were read there rather than inferred from a different TimeNet release.

| Source | Relevant contract |
|---|---|
| [AGENTS.md](https://github.com/OpenTSLM/TimeNet/blob/main/AGENTS.md), [CONTRIBUTING.md](https://github.com/OpenTSLM/TimeNet/blob/main/CONTRIBUTING.md) | uv workflows, Python support, mandatory checks, MIT contributions, no synthetic data repair |
| [Architecture](https://github.com/OpenTSLM/TimeNet/blob/main/docs/architecture.md), [connectors](https://github.com/OpenTSLM/TimeNet/blob/main/docs/connectors.md) | producer/consumer separation, task-level splits, per-connector isolation |
| [BaseConnector](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/connectors/base.py) | `BaseConnector[TRaw]`; no-argument construction; `download(cache_dir)` or `download_async`; `convert(raw_refs)` or `compose`; concrete `metadata()` loads adjacent `dataset.yaml`; `CARD` override; `values_backend` defaults to Parquet |
| [Dataset](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/dataset/dataset.py) | `TimeFDataset(metadata=...)`, `add_record(record=...)`, `add_task(task=...)`, `derive_schema()`, `write(path=..., values_backend=...)`, `get_train()`, `get_validation()`, `get_test()`, `iter_tasks(split=...)` |
| [Record](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/dataset/record.py), [Source](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/dataset/source.py) | `Record(sources=..., start_time=..., metadata=..., time_span=...)`; `Source(name=..., signals=..., metadata=...)`; `.annotate()` returns a bound annotation occurrence |
| [Signal implementation](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/dataset/time_series.py) | `Signal.from_values(values, *, spec, name, time_axis)`; `Signal.from_irregular(values, *, time_offsets_us, spec, name, full_validation=True)`; `from_loader` for lazy values; shape has leading time dimension |
| [Specs](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/specs.py) | `TimeSeriesSpec(spec_type, name, unit_value, dtype='float32', categories=(), value_shape=(), dimension_names=(), missing=None)`; scalar shape default; enum values are category strings |
| [Axes](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/dataset/axis.py), [clock](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/clock.py) | `RegularAxis(period_us=Fraction(...))`, `OrdinalAxis()`, `IrregularAxis`; integer microsecond offsets; `TimeOrigin(timestamp=None)` means no known Unix anchor |
| [Annotations](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/annotations.py), [spans](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/spans.py) | `Annotation(key=..., value=..., span=..., metadata=...)`; `TimeInterval.micros(start,end)`; static/point/interval annotations; annotations reject StepSpan |
| [Tasks](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/tasks.py), [label sets](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/label_sets.py) | `TemporalLocalizationTask`, `LocalizationMode`, `ClassificationTask`, `Label`, `LabelSet`; closed built-in task classes, not extension through custom subclasses |
| [Splits](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/splits.py) | `Split.TRAIN`, `Split.VALIDATION`, `Split.TEST`; values `train`, `validation`, `test` |
| [Strict JSON](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/json.py) | `JsonMapping` accepts nested JSON-native values; reject NumPy objects, nonfinite floats; convert arrays with dtype/shape descriptors and native lists |
| [Metadata](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet/src/timenet/types/metadata.py), [manifest](https://github.com/OpenTSLM/TimeNet/blob/main/docs/manifest.md) | `DatasetMetadata.from_yaml`; card version, source revision, license, access; TimeF format version 2, distinct from dataset/card versions |
| [Discovery](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/discovery.py) | lazy package discovery, `CONNECTOR`, adjacent card; no central registration table |
| [Build CLI](https://github.com/OpenTSLM/TimeNet/blob/main/docs/cli/build.md) | exact `timenet-build build` options, local registry, cache retention, immutable publishing |

### Reference connectors

[hello_world/connector.py](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/datasets/timenet/hello_world/connector.py)
is a deterministic offline BaseConnector example. It creates shared immutable
specs/axes, separately owned Signals, Sources and Records, adds Records before
Tasks, and uses `.annotate()` return values in `target_annotations`. Its scoped
ClassificationTask, interval annotations, and sparse localization are directly
relevant. Do not reuse its synthetic wave generator or 16 Hz cadence.
Its [tests](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/datasets/timenet/hello_world/tests/test_connector.py)
check deterministic conversion, ownership, schema features, and task chains.

[tsqa/connector.py](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/connector.py)
uses BaseHuggingFaceConnector, pinned `SOURCE_REVISION`, declared source files,
one Signal per channel, and explicit `task.split`. Its AnswerTask is grounded in
actual source QA text. KernelSight has no such QA or explanation targets, so do
not invent rationales or copy the QA task pattern merely to fit the example.
TSQA's **generated** 80/10/10 StratifiedGroupKFold partition is unsuitable here:
KernelSight already has official partitions. Its [tests](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/tests/test_connector.py)
provide synthetic offline fixtures and stored-split round-trip examples using
`LocalRegistry.put_dataset`, `open_local_version`, and `TimeFReader`.

## 3. Field-by-field mapping

Canonical means reconstructible exactly from the release with a documented
inverse decoder, not only visible in a display. Recommend a namespaced
`record.metadata['kernelsight_v4']` JSON payload with `schema_version: 1` and
array descriptors `{dtype, shape, values}`. This is a connector schema, not an
existing TimeNet class. Convert NumPy scalars/arrays to native values explicitly.
Retain integer nanoseconds as integers; never pass through floating seconds.

| KernelSight field | Proposed TimeF representation | Preservation and use |
|---|---|---|
| `data`, float32 `[24,512]` | 24 `Signal.from_irregular(data[c], ...)` under a single counter Source; common scalar float32 spec | Preserve channel sequence and values; decoder stacks Signals by explicit original `channel_index`; no rescale/transpose of each 1-D channel |
| `counter_names[24]` | original unique `Signal.name` plus ordered canonical list and per-Signal metadata index | Do not rename to `c0` or sort alphabetically; reject duplicates rather than silently rename |
| `time_edges_ns[513]` | canonical int64 array in Record metadata | Exact original origin and final right edge retained; compare input and label copies before storing one common array |
| derived placement of 512 values | `from_irregular(..., time_offsets_us=(edges[:-1]-edges[0])//1000, ...)` | Explicit left-bin timestamp convention; derived floor quantization, not exact native ns support |
| no known calendar anchor | default `TimeOrigin()` with `timestamp=None` | Do not interpret GPU/profiler clock readings as Unix nanoseconds |
| `workload_l1[512]`, `workload_l2[512]` | canonical int32 arrays; optional derived annotation runs with class text and numeric ID metadata | Preserve -1 and class ordering; runs must not merge across unlabeled gaps |
| L1/L2 multihot `[512,12]` / `[512,73]` | canonical uint8 arrays; interval annotations per class's contiguous active runs | Never argmax, choose a dominant class, or flatten multiple active classes |
| `mask_labeled[512]` | canonical uint8 array | Preserve original; assert agreement with `workload_l1>=0`; do not substitute activity |
| L1/L2 valid masks | derived separately from each primary array, optionally materialized in payload with provenance | No assumption of identical missingness across hierarchies |
| `mask_any_kernel[512]` | canonical uint8 array | Independent kernel-activity track; not a primary supervision mask |
| `multihot_n_active[512]`, `multihot_has_overlap[]` | canonical original values plus independent derived consistency checks | L1 active count and >=2 definition; same-class concurrency remains outside this target |
| `segment_id[512]` | canonical int32 array | Preserve -1 and original segment ordinals |
| `segment_starts`, `segment_ends` | canonical int64 bin indexes, inclusive/exclusive; derived interval annotations | These are **bin indexes**, not ns; native interval uses edges at those indexes |
| `segment_label_l1/l2`, `segment_kernel_names` | canonical arrays/text lists; annotation occurrence metadata for segment index and IDs | Keep segment labels distinct from dense labels and derived per-class runs |
| `segment_predecessor_l1/l2`, `segment_position` | canonical original integer arrays | Do not recompute from reordered or merged intervals |
| `attribute_flags[S,8]`, `attribute_flag_names[8]` | canonical array and ordered names; optional descriptive annotations | Target/annotation metadata, never automatic model features |
| `vocab_l1[12]`, `vocab_l2[73]` | full ordered canonical text lists | Include absent/unused entries, including unobserved primary classes |
| `l2_parent_l1[73]` | canonical integer mapping | Full hierarchy, not inferred from string splitting |
| `spatial_state_vocab[5]` | canonical ordered text list | No fabricated per-bin spatial-state labels |
| input `kernels[K,2]`, `kernel_names[K]`, `kernel_function_index[K]` | canonical launch arrays/text; optional Source provenance metadata | Keep launch timestamps and function IDs exact; identity-bearing metadata excluded from prompts/features |
| fingerprint NPZ | canonical namespaced auxiliary payload with every actual key, dtype and shape | Not model input; do not assume one `[32]` vector per trace |
| official split record | canonical original JSON record, path, source split, index within split; `task.split` mapping | Split preserved at trace level across every task family |
| source identity/provenance | Source describes NVIDIA H100 / profiling sources; dataset-level static provenance annotation | Record NPZ/split hashes, source commit, hardware, collector limitations, conversion schema/version |

The observed training fingerprint example contains `fingerprint[6,32]`,
`class_names[16]`, `pipe_names[16]`, `kernel_ids[6]`, and `kernel_names[6]`.
Thus the short “32-D fingerprint” description is not a sufficient lossless
serialization specification. Inspect fields during conversion and reject unknown
unsupported fields rather than discard them. The same rule applies to NPZ schema
changes. Original archives remain external raw sources pinned by hash; preserving
array content does not promise byte-identical reconstruction of NPZ compression.

A common counter spec can be constructed as
`TimeSeriesSpec(spec_type='kernelsight_counter', name='Physically scaled GPU counter',
unit_value=ureg.dimensionless, dtype='float32')`, importing `ureg` from
`timenet.types`. Store upstream divisors/group descriptions as provenance rather
than pretend every value is an unscaled percentage. A single tensor-valued Signal
is also supported by `value_shape=(24,)` with time-major `[512,24]`, but 24 scalar
Signals better preserve names and heterogeneous counter meanings and keep the
Parquet path simple. No need for Zarr or an additional input modality in v1.

## 4. Time precision, intervals, unlabeled bins, and overlap

### Native timeline limitation is real

Read-only inspection of the first official training trace,
`kernels/cutlass_fmha/_out/b1h32/input/tensor_input.npz`, found first/last edges
500545205 / 613957088 ns. Integer bin widths are 221507 or 221508 ns; 512 of
513 absolute edges have a nonzero remainder modulo 1000. A single uniform
integer-nanosecond period is therefore not exactly the saved grid.

TimeF IrregularAxis requires one int64 **microsecond offset per value**, not
513 edges for 512 values. RegularAxis stores a Fraction period but resolves
positions to whole microseconds, requires period >=1 us, and cannot reproduce
an arbitrary saved edge sequence. `TimeInterval.seconds` rounds to microseconds;
using float seconds does not solve this. OrdinalAxis preserves only order, and
StepInterval is not a time interval: Annotation explicitly rejects StepSpan.

Recommendation: retain exact edges canonically; derive left offsets by floor
of relative ns/1000 and intervals from the same edge convention. Declare
`Record.time_span=TimeInterval.micros(0, floor((edges[-1]-edges[0])/1000))`
when valid so final-bin annotations fit the declared recording extent, not
merely the last stored sample. Validate that it covers every Signal window.
Require noncollapsed intervals and valid increasing source edges. If a native
interval collapses after quantization or record extent fails validation, raise
an error and resolve the policy; do not stretch, interpolate, or silently repair
it. Exact bin indexes and ns bounds remain authoritative. Native coordinates
have <1 us floor error at each edge, not nanosecond fidelity.

If exact native ns localization is a hard requirement, it requires an upstream
format/API discussion and likely a versioned format change; a connector alone
cannot provide it. Ordinal-only storage plus exact edges is another possible
canonical choice, but sacrifices ordinary native time-annotation support.

### Unlabeled is not an operation or negative evidence

Preserve -1 in signed canonical arrays and masks. Do not add “unlabeled” to the
12/73 operation vocabularies, map it to `other`, or cast it to unsigned IDs.
TimeSeriesSpec enum Signals take category text, not arbitrary signed ID arrays;
converting -1 to an enum index is incorrect. If a future nullable categorical
Signal view is desired, use `MissingValues(reason=MissingReason.NOT_RECORDED,
description=...)` and Arrow nulls, retaining original -1 arrays/masks for inversion.
It must remain outside model input Sources. NumPy conversion of null-bearing
Signals is restricted, so this is not recommended as the initial representation.

`LabelSet(name=..., labels=tuple(Label(name=v) for v in vocab), multi_label=False)`
can represent the complete vocabulary for future ClassificationTasks. Construct
separate multi-label sets with `multi_label=True`. Primary ClassificationTasks
require exactly one valid textual class: skip unlabeled bins rather than supply
`('-1',)` or an empty single-label target. An empty multi-label target means “no
label applies” in TimeNet; it is not synonymous with unknown or unsupervised.
Only generate such negatives after explicitly agreeing on supervision coverage.
Label sets enter the dataset schema through classification tasks, not merely by
constructing LabelSet objects. Our localization-only v1 must retain full vocab
and parent mapping in the canonical payload even if schema has no label sets.

### Overlap and localization tasks

For every L1 class, find contiguous runs where its multi-hot column equals 1.
Different class runs may overlap; emit them independently. Attach each
`Annotation(key='kernelsight_l1_active', value=class_name, span=TimeInterval.micros(...),
metadata=...)` to its Record, with exact start/stop bins and ns bounds in metadata.
Use the returned annotation occurrences as a localization task's
`target_annotations`, not as `input_annotations`. Keep original source segments
under a separate key; they are not interchangeable with these runs.

Add the Record to TimeFDataset before adding its task. Proposed task construction
uses `TemporalLocalizationTask(inputs=(record,), prompt=<fixed localization query>,
mode=LocalizationMode.SPARSE, target_annotations=tuple(bound_occurrences),
split=official_split)`. Do not also set inline `targets`; the Dataset rejects
simultaneous inline and annotation answers. For a trace with no positive runs,
set `targets=()` instead of absent targets and absent annotations; verify the
empty-output round trip. No classification of unknown bins is implied.

Prefer SPARSE initially: labels do not establish exhaustive negative coverage
of all bins. If EXHAUSTIVE is desired, define a supervised scope and its validity
semantics first. No QA text, explanation, rationale, or causal claim is generated.
Consumers must select counter values only; generic access to a Record also
exposes labels in metadata and attached target annotations. A leakage audit of
the eventual model adapter is necessary, even with empty `input_annotations`.

## 5. Preserve official partitions exactly

Read only the existing `train.json`, `val.json`, and `test.json` as partition
authority, retaining source order and original path strings. Map:

| Source | TimeF task split | Unique Record count |
|---|---|---:|
| `train` | `Split.TRAIN` | 1,124 |
| `val` | `Split.VALIDATION` | 160 |
| `test` | `Split.TEST` | 160 |

No splitter, seed-based reassignment, category balancing, or deduplication that
merges distinct source traces. Assert unique and pairwise-disjoint path lists,
complete coverage, and no Record referenced by tasks from different partitions.
Preserve split JSON hashes and trace index provenance. One proposed task per
trace yields the same task counts as Record counts. With additional task
families, assert these counts **per family**, and separately count unique Records;
the aggregate number of tasks will be larger than 1,444.

Store the official partition on each task and as record provenance, since Record
has no native `split` field. Use reader/SDK stored split accessors, never rerun
splitting when loading. `iid`, `param_ood`, `composed`, and `length_ood` may be
retained as analysis-tag membership in metadata; they are not independent splits.
Path disjointness is not kernel-identity independence; retain that limitation.

Task 2 development should use synthetic cases and real training traces only.
The eventual complete release necessarily reads held-out bytes to serialize them;
separate that mechanical conversion/integrity audit from development inspection,
model selection, prompt design, or fitting. Agree on that evaluation protocol
before executing an all-partition build. This research task opened no held-out
NPZs.

## 6. Registration, source acquisition, versioning, licensing

Proposed upstream location (not created in this task):

```text
packages/timenet-connectors/src/timenet_connectors/datasets/
  williamhtan/
    __init__.py
    kernelsight/
      __init__.py             # exports CONNECTOR and connector class
      connector.py            # BaseConnector subclass, download/convert
      dataset.yaml            # card and release identity
      requirements.txt        # independent source dependencies
      source.py               # raw refs, pinned acquisition, strict NPZ reads
      mapping.py              # lossless payload and derived temporal views
      tasks.py                # task builders with explicit official split
      README.md               # schema, provenance, inverse-decoder contract
      tests/
        __init__.py
        test_connector.py
        test_mapping.py
        test_roundtrip.py
        fixtures/             # synthetic NPZ/JSON only
```

`discovery._module_name` maps user hyphens to Python underscores, as in
`timenet/hello-world -> timenet.hello_world`. There is no central registry list.
The org package must exist, the connector package exposes `CONNECTOR`, and
`metadata().dataset_id` must match the requested ID. Do not appropriate the
`opentslm` namespace without agreement. An alternative research-owned namespace
is possible, but choose it before publishing to avoid identity churn.

Recommend BaseConnector with a frozen lightweight raw-ref dataclass containing
local root, official split paths, and source revision. Read configuration from
environment in no-argument `__init__`, e.g. a proposed `KERNELSIGHT_ROOT` for a
read-only local source. This environment variable is our proposed interface,
not an existing TimeNet setting. `download` only locates/fetches files;
`convert` reads validated references offline. Do not depend on Kernel X-ray's
PyTorch loader, its local path aliases, or its Python 3.9 environment.

For remote acquisition, use `hub_snapshot` from
[timenet_connectors.sources.huggingface_hub](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/sources/huggingface_hub.py)
inside `download_async`, with `williamhtan/kernelsight`, a **40-character pinned
commit**, and explicit patterns for NPZs and split JSONs. Pinning that revision
is unresolved: the downloaded dataset's name `v4` is not an immutable commit.
Verify downloaded arrays/split hashes against the current local dataset before
claiming equivalence. Include upstream documentation/license evidence too.
Alternatively BaseHuggingFaceConnector supports `HF_REPO`, `SOURCE_REVISION`, and
`SOURCE_FILES: part -> path/pattern`; its HubFileRef/check_refs/ref_paths API is
verified in [bases/huggingface.py](https://github.com/OpenTSLM/TimeNet/blob/main/packages/timenet-connectors/src/timenet_connectors/bases/huggingface.py).
Avoid pretending it accepts a repository root with no declared parts. The custom
BaseConnector route better accommodates the initial local-root mode.

Declare all source dependencies in this connector's own requirements file,
including compatible pinned NumPy and Hugging Face Hub versions when used;
lazy-import optional download dependencies. Do not inherit a sibling connector's
requirements or pull PyTorch merely to decode NPZ arrays. Source helpers support
cache and HF_TOKEN; do not embed credentials. Non-Hub file URLs require immutable
URLs and SHA-256 via `Artifact`/download helpers. Never point TimeNet's disposable
download cache directly at the user's dataset, or return lazy loaders whose raw
sources vanish before writing. Engine cleanup removes downloaded raw cache on
success unless `--keep-cache`; external local input is read-only and never
engine-owned. Use eager arrays initially or ensure lazy lifetime through storage.

Card fields: `yaml_schema_version: 1`, chosen `dataset_id`,
`dataset_version: 1.0.0`, exact `source_revision`, name/description, source URL,
`license: Apache-2.0`, `access: open`, citation, and valid domains/tags. The source
card claims Apache-2.0 and attributes KernelBench MIT and CUTLASS BSD-3-Clause;
retain notices/provenance. TimeNet connector code contributions are MIT, which
is separate from the data license. Confirm actual pinned-source license material
and redistribution permission before publishing; a card's claim is evidence,
not a substitute for retaining required notices. Run upstream `make license-check`
for dependency policy (it rejects copyleft dependencies).

Bump the compiled dataset version for changes to tasks, split interpretation,
canonical schema, timing policy, or normalization. Keep source revision separate.
TimeF format version 2 and card schema version 1 are unrelated to these numbers.
Remote publishing also requires clean committed TimeNet package provenance.
Task 2 should produce local artifacts only; publishing is a separate action.

## 7. Reproducibility, build procedure, and tests

TimeNet requires Python >=3.11 and its checkout selects 3.13. Kernel X-ray's
working `.venv` is Python 3.9.6. **Do not upgrade or replace that environment**.
Use a separate TimeNet uv environment under the upstream workflow when Task 2
is authorized. No dependencies or TimeNet code were executed/installed here.

Future commands after the connector exists, not commands run in Task 1:

```bash
# In the TimeNet development checkout, using its separate uv environment:
make sync
make check
make lint-fix
make test
make test-connectors
make license-check

# Environment variable below is a proposed connector option.
KERNELSIGHT_ROOT="/path/to/kernel-xray/data/kernelsight_v4" \
  uv run timenet-build build williamhtan/kernelsight@1.0.0 \
  --keep-cache --local-registry "/path/to/kernel-xray/outputs/phase2/timef"
```

The final command is the eventual full conversion and is gated by the held-out
protocol above. Initial smoke conversion should directly call `convert` with
synthetic/training fixture refs into a temporary registry, not publish a
training-only artifact under the full official release identity. Isolation is on
by default; `--no-isolation` is only for an explicitly prepared development env.
`--dry-run` can inspect the build plan but does not establish data correctness.
For CI equivalence, also use `make check-ci` and `make test-unit`; the former
includes Python 3.11 typing in addition to the workspace's 3.13 target.

Required automated validation before accepting Task 2:

1. **Card/discovery/offline conversion:** metadata ID/version/license match;
   resolve only the requested connector; instantiate without arguments; synthetic
   `convert` performs no network access; missing files fail with TimeNet errors.
2. **Every field round trip:** compare raw counters bitwise in float32, 24 channel
   names/order, 512 steps, all 513 int64 edges, original labels/masks/multi-hot
   tracks, all segment/launch fields, auxiliary fingerprint keys/shapes/dtypes,
   vocabularies and parents after inverse decoding. Check source hashes first.
3. **Timeline:** nonzero ns remainders, alternating integer bin widths, final
   right edge, non-Unix clock, sub-us/collapsed intervals, bounds and record extent.
   Exact canonical edges must match; derived native offsets must follow the
   declared floor rule and expose their loss, not pass with an arbitrary tolerance.
4. **Labels:** -1 in both primary tracks preserved; class IDs in valid range;
   primary IDs included in multi-hot where valid; masks remain independent;
   all-zero, single-class, and multi-class bins; transitions at first/final bins;
   discontiguous runs and overlapping intervals; all 12/73 entries including
   labels absent from observed primary targets; hierarchy and source counts agree.
5. **Tasks/splits:** annotations are bound occurrences; Record registered first;
   targets never also appear in input_annotations; target answer representation is
   unambiguous; empty positive output supported; every task has mapped official
   split; no cross-split Record reuse; counts 1,124/160/160 per family; unchanged
   official paths/hashes. Analysis tags cannot become partitions.
6. **No leakage or fabricated supervision:** model-facing adapter emits exactly
   the 24 counters; filenames, annotations, canonical target payload, kernel
   identities and masks do not appear in prompts/features; no invented explanation
   targets or negative labels for unknown bins. No normalization during conversion.
   Any later preprocessing uses only the saved training population.
7. **Actual storage:** use `LocalRegistry.put_dataset(original)` and reopen through
   `TimeFReader(open_local_version(version_dir))` as TSQA tests do; verify values,
   metadata, annotation refs, task splits, derived schema and manifest checksums.
   In-memory success alone is insufficient. Test Parquet; test Zarr only if chosen.
8. **Determinism/integrity:** repeated conversion with stable official ordering
   yields equivalent semantic datasets and canonical hashes; record source commit,
   all input/split SHA-256, source release, conversion schema, TimeNet commit,
   dependencies and build environment. Do not promise byte-identical DuckDB files
   without measuring it. Hash raw files, splits and completed outputs before/after.
9. **Evaluation boundary:** synthetic fixtures exercise all partitions; real
   development fixtures are training-only, with a read guard against held-out
   arrays. A separately authorized full-release integrity job may serialize and
   compare held-out bytes mechanically, never tune mapping based on their content.
10. **Existing project regression:** rerun Kernel X-ray's 32 tests and completed
    visualization verification without regenerating plots. Keep TimeNet's tests
    in its own environment and report checks separately.

Use `TimeNetConnectorError` for invalid source assumptions, `TimeFValidationError`
for bad model invariants, and `TimeFFormatError` for corrupt/unsupported artifacts;
never repair or fill missing data silently. Follow upstream type hints,
Google-style docstrings, ruff/ty rules, and normal pre-commit checks.

## 8. Decisions to resolve before implementation

| Decision | Recommendation / outstanding evidence |
|---|---|
| Native ns accuracy | Accept exact canonical ns plus explicitly quantized native temporal view; otherwise discuss a TimeF extension before connector work |
| Dense-target representation | Approve JSON-native versioned metadata payload with inverse decoder; typed target Signals would need a safe non-input design and extra ownership/tasks decisions |
| Task contract | Start with one sparse multi-hot L1 localization task per trace; decide whether Task 2 also needs primary L1/L2 classification or L2 localization and define task counts/coverage |
| Negative/unlabeled coverage | Keep unknown separate; do not choose exhaustive localization or empty multi-label negatives without a supervision protocol |
| Dataset identity | Confirm `williamhtan/kernelsight` namespace or select an authorized research namespace before registration |
| Immutable source revision | Identify the HF commit matching current local NPZs and split hashes; do not guess from download time or cache directory names |
| License evidence | Confirm pinned source license/notices and retain upstream attribution before redistribution |
| Held-out conversion boundary | Authorize a mechanical full build/audit separately from training-only development; no inspection for heuristic/prompt decisions |
| Environment and task location | Separate Python >=3.11 uv environment; future upstream changes in an authorized development checkout/worktree; preserve current 3.9.6 project env |

## 9. Phase 2 Task 2 implementation sequence

1. Resolve the table above, freeze the conversion/task schema and acceptance
   criteria, identify/pin the matching source revision, and capture baseline hashes.
2. Prepare an authorized TimeNet development checkout and separate uv environment;
   leave the current research checkout/data/visualizations intact.
3. Add the proposed namespace/package, card, independent requirements, raw-reference
   type, and strict offline parser. Test registration with tiny synthetic inputs.
4. Implement one Record/24 Signals per trace, exact canonical payload encoding and
   inverse decoding. No training, normalization, or task heuristics at this stage.
5. Implement the agreed native timeline view and fail clearly on unsupported
   precision/bounds. Preserve original segments separately from derived runs.
6. Build agreed task families with attached target occurrences and explicit
   official split translation. Verify no label exposure through model inputs.
7. Write synthetic edge-case tests and training-only real fixtures; verify an
   actual temporary-registry write/read round trip and deterministic reconstruction.
8. Run TimeNet contributor checks and existing Kernel X-ray regressions. Fix source
   mapping errors without regenerating official partitions or repairing source data.
9. With the held-out protocol authorized, perform one mechanical full conversion,
   validate unique trace counts and per-family split counts, compare canonical
   arrays/hashes, and save a provenance/validation report under outputs/reports.
10. Review the local release and its documented precision/coverage limits. Commit
    under upstream conventions if requested. Publishing or model training remains
    outside Task 2 unless separately requested.

## 10. What was verified in this research task

Read local upstream instructions, connector docs, architecture/build/manifest
specification, core object source, discovery and Hub base implementation, both
reference connectors and their tests, and local KernelSight schema documentation.
Read one real training input/fingerprint pair to check timing precision and
auxiliary shapes. No real validation/test NPZs were opened. No connector, build,
model, or runtime API compatibility test was run; the APIs above are verified by
static source inspection. Upstream contributor checks were not run because this
was a read-only design task with no upstream changes. The proposed mapping and
its inverse still require the tests listed above before claiming a working or
lossless TimeF integration.
