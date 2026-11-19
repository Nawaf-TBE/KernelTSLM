# TimeNet technical decision review

18 November 2026, Asia/Muscat. This review supersedes recommendations in
`timenet_integration_plan.md` where they differ. No connector is implemented.
TimeNet source inspected: `9bd1613019ee61fbc52e02106a1a45990ead7a54` (clean checkout,
package 0.12.0). Source snapshot verification and licensing are reported separately
in [source_provenance_and_license.md](source_provenance_and_license.md).

## Method and evidence classification

**Hard requirements:** reconstruct required source arrays exactly, preserve official
trace partitions, and keep targets/identity information out of model features.
Reject any alternative that violates these, regardless of convenience. Among
remaining alternatives compare fidelity, TimeNet compatibility, research validity,
interoperability, reproducibility, maintenance, and performance. No weighted score
can override a hard requirement. A supported API is not proof of a working
conversion; storage round trips and model-input audits remain acceptance gates.

Labels used below: **Fact** = inspected implementation/documentation; **Observation**
= local dataset or hash result; **Judgment** = engineering recommendation;
**Unresolved** = evidence/approval still needed. Alternatives are not all equivalent
on interoperability or resource costs, even when all could preserve data.

### Source anchors

Paths below are relative to `../TimeNet/packages/timenet/src/timenet/` unless
otherwise stated. Line numbers refer to this checkout, not future releases.

| ID | Actual source and symbols |
|---|---|
| T1 | `dataset/axis.py:62` RegularAxis, `:266` IrregularAxis, `:331` OrdinalAxis; `types/clock.py:21` TimeOrigin; integer microseconds, Fraction cadence >=1 us |
| T2 | `dataset/time_series.py:219` Signal, `:430` span_us, `:456` from_values, `:492` from_irregular; irregular window ends one us after last sample |
| T3 | `types/specs.py` TimeSeriesSpec, value_shape, dimension_names, MissingValues; `values_backends/parquet/writer.py:154` rejects nonempty value_shape and directs callers to Zarr |
| T4 | `types/annotations.py:62` Annotation and __post_init__: scalar/text values, JSON metadata, only TimeSpan; `types/spans.py:197` TimeInterval, `:297` StepInterval; `dataset/span_validation.py` check_span_within_window |
| T5 | `types/tasks.py:139` Task, `:374` ClassificationTask, `:461` TemporalLocalizationTask, `:620` _is_target_item; `types/label_sets.py:22` LabelSet, check_targets |
| T6 | `dataset/dataset.py:332` add_task, `:388` set_task_stream, `:788` _check_task_answer, `:807` _check_annotation_refs; stream factory must be reiterable and cannot mix with add_task |
| T7 | `torch.py:51` TimeFTorchDataset, `:163` _item: emits ALL record.signals as series, plus tasks and annotations; masks are all true for nonmissing values; nulls rejected |
| T8 | `types/splits.py` Split; `types/metadata.py:38` DatasetMetadata; `json.py` strict JsonMapping; `dataset/record.py` Record.metadata/time_span |
| T9 | sibling `timenet-connectors/src/timenet_connectors/discovery.py` connector_dir/resolve; `bases/huggingface.py` BaseHuggingFaceConnector, HubFileRef; `sources/huggingface_hub.py` hub_snapshot |
| T10 | TimeNet `AGENTS.md`, `CONTRIBUTING.md`, `docs/connectors.md`, `docs/cli/build.md`, `docs/manifest.md`, `docs/timef-writer.md`: uv, Python >=3.11, release/provenance and check requirements |

Reviewed implementation references:
[hello_world connector](../../TimeNet/packages/timenet-connectors/src/timenet_connectors/datasets/timenet/hello_world/connector.py)
shows owned Signals, reusable specs, scoped classification, bound target annotation
occurrences and sparse localization. Its tests check deterministic conversion.
[TSQA connector](../../TimeNet/packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/connector.py)
shows channel Signals, pinned source, task splits and QA. Its tests at lines
186–197 reopen `LocalRegistry.put_dataset` output with TimeFReader and check splits.
TSQA generates splits; KernelSight must not copy that part. Existing connector
QA/rationale content does not justify synthesizing KernelSight explanations.

## A. Timestamp decision record

| Field | Review |
|---|---|
| Decision / requirement | Preserve 513 int64 ns boundaries and bin identity exactly, while supporting native temporal localization |
| Alternatives | (1) Microsecond native timeline plus exact edges; (2) OrdinalAxis plus exact edges; (3) RegularAxis with Fraction period plus edges; (4) separate numeric edge Signal, ordinal, not a physical time axis; (5) upstream ns schema extension |
| Evidence | T1/T2/T4; original data_info §3 and segment schema; training-only audit below |
| Hard constraints | No dropped final edge, fabricated Unix anchor, floating-point ns conversion, reordered bins, or silent rounding of canonical times |
| Feasibility | 1–4 are supported, with different semantics; 5 requires upstream implementation, format/version review and authorization |
| Trade-offs | 1 supports standard time spans but quantizes native coordinates; 2 preserves order but cannot attach normal temporal annotations; 3 can encode a rational average cadence but not the exact alternating saved integer-ns widths; 4 stores ns as ordinary values, not native temporal semantics, and must not pollute inputs; 5 gives native ns semantics at ecosystem-wide cost |
| Recommendation | **Judgment:** retain exact original edges in canonical payload; use 512 relative left-edge floor-us offsets with from_irregular for counter Signals and native tasks |
| Justification | Exact source reconstruction and interoperable temporal views can coexist if native precision loss is declared. Ordinal-only loses useful localization APIs; a core change is disproportionate when exact boundaries remain accessible |
| Verification | Exact int64 edge round trip including 513th edge; q_i=floor((e_i-e_0)/1000); check 0<=e_i-e_0-1000*q_i<=999 ns; monotonic q; validate final Record.time_span; test negative origins, very large ns, irregular widths, duplicate/collapsed q, and final-bin tasks |
| Confidence | High on source limitation; Medium on final mapping pending real writer/reader and task-scope round trips |
| Open questions | Do consumers require native ns precision rather than exact auxiliary ns? If yes, stop and seek core support; never label the native view lossless |

**Observation:** across all **1,124 training traces**, maximum floor residual is
999 ns and minimum quantized bin width is 200 us. No training bin collapses.
For each endpoint loss is less than 1 us; interval duration error is the difference
of endpoint errors and has absolute magnitude below 1 us. Floor shifts native
edges earlier; it is not symmetric rounding. These measured extrema do not claim
the same extrema for held-out tensors, whose contents were not decoded here.

Record.time_span must cover `[q_0,q_512)`, because IrregularAxis.span_us otherwise
ends at `q_511+1`, which is not the last bin's right edge. Use unscoped whole-record
time spans for tasks that need the last bin: a scope explicitly naming Signals
may instead be checked against their shorter irregular windows. Test this
interaction, not only timestamp arithmetic. A collapsed interval must fail; no
padding or duration inflation. GPU timestamps are not proven Unix timestamps:
TimeOrigin.timestamp stays None and original e_0 is preserved separately.

## B. Counter Signal decision record

| Field | Review |
|---|---|
| Decision / requirement | Preserve 24 counter meanings/order and 512 float32 samples with efficient standard access |
| Alternatives | 24 scalar Signals; one tensor Signal [512,24] with value_shape=(24,); 24 single-Signal Sources; different Records per channel |
| Evidence | T2/T3/T7; TSQA loops over channels; hello-world separate owned Signals; upstream physically scaled counters |
| Hard constraints | Exact dtype/values, stable names/order; no target channels or silent feature transformations |
| Feasibility | Scalar Signals work in Parquet; tensor Signal is supported with Zarr, not Parquet. Separate Sources/Records are possible but complicate ownership/alignment |
| Trade-offs | Scalars add 34,656 counter Signal objects but expose channels naturally and avoid Zarr dependency. Tensor reduces descriptors and makes whole-trace access convenient, but needs ordered channel metadata, transpose/inverse, Zarr and tensor compatibility tests. Single tensor can preserve names if metadata is explicit; it is not intrinsically lossy |
| Recommendation | **Judgment:** one trace Record, one counter Source, 24 scalar float32 Signals, Parquet |
| Justification | All counter series are short and synchronized; native names/per-counter metadata and simpler build/read path outweigh unmeasured tensor speed advantage. Source grouping is not a claim that all counters came from one physical collector |
| Verification | Bitwise float32 comparisons, signed zero retention, original names/order, dictionary/name selection, exact inverse stack to [24,512], all source/owner counts, per-channel and full-trace reads, dtype-preserving Parquet round trip |
| Confidence | High compatibility; Medium performance choice because no converted-storage benchmark has run |
| Open questions | Benchmark tensor/Zarr only if scalar overhead is material; do not assume it is faster. Agree timing precision as in A |

No additional z-score, display p99, interpolation, clipping or augmentation during
conversion. Values already have upstream physical scaling. Use dimensionless
value units with documented counter-specific normalization provenance, not guessed
percentages. Preserve original counter names and channel indexes explicitly even
if a reader returns Signals in a different traversal order.

## C. Dense supervision decision record

| Field | Review |
|---|---|
| Decision / requirement | Preserve primary labels/-1/masks, both multi-hot hierarchies, segment arrays, all 12/73 vocab entries and parents; make supervision meaningful to ordinary TimeNet tooling |
| Alternatives | JSON-only metadata; interval annotations only; per-bin classification only; typed numeric/enum label Signals on input Record; typed target Signals/Record on separate target owner; hybrid native tasks/annotations plus exact payload |
| Evidence | T3–T8; targets accept Signal/Record objects (tasks.py:28–35,620), not arbitrary NumPy arrays; Annotation.value does not accept numeric arrays; Annotation.metadata accepts nested JSON |
| Hard constraints | -1 stays unknown, never other; no argmax of multi-hot; every original field round-trips; target features never enter inputs |
| Feasibility | JSON-only preserves but lacks native supervision semantics. Annotations/tasks alone need masks/exact indexes to recover unknowns and ns. Signed typed target Signals are feasible; nullable enum is also possible but nulls fail standard torch conversion. Putting them on an input Record is unsafe: T7 includes all Signals |
| Trade-offs | Typed target storage is more structured/efficient for large dense arrays but needs extra owner Records and careful input-only selection; a generic tensor-output task is not automatically a standard operation-classification task. Per-column scalar multi-hot adds 85 Signals/trace; matrix Signals require Zarr. Metadata is simple but untyped, JSON-heavy and not automatically exposed by TimeFTorchDataset._item |
| Recommendation | **Judgment:** hybrid: native primary classification and native multi-hot localization at both hierarchies, plus exact versioned payload for inverse reconstruction. Do NOT expose dense labels only in metadata |
| Justification | Standard tasks become usable without a KernelSight-specific dense decoder, while payload retains -1, exact bins, masks, source segments and auxiliary fields. This corrects the original metadata-heavy/L1-only recommendation |
| Verification | Decode all fields/dtypes/shapes; reconstruct each valid bin from native tasks plus exact indices; validate unknown coverage against canonical masks; full ordered LabelSets with unused classes; parent consistency; multi-hot overlap round trip; raw vs task target names/IDs agree |
| Confidence | Medium: APIs support the design; control-plane size, serialization cost and actual scope behavior must be tested |
| Open questions | Upstream maintainers may prefer typed target records; seek feedback on metadata volume and dense-target conventions before claiming upstream acceptance |

Canonical payload uses namespaced Record.metadata with dtype/shape/native JSON
values. Arrays, original segments, auxiliary fingerprint contents and exact ns
retain their source meanings. Store all source keys, including zero-observed vocab
entries, and fail on unsupported schema additions. No unsigned coercion of -1.
A dtype='enum' Signal consumes strings; it does not directly encode negative
integer class IDs. Nullable categorical convenience views need MissingValues and
explicit masks; they cannot replace original signed arrays.

Vocabularies are full ordered LabelSets for native primary classification, with
separate hierarchy names. Native localization annotations carry original class
text plus numeric ID/exact bin bounds. Preserve the L2 parent mapping separately;
LabelSet does not encode this hierarchy. Multi-label LabelSets are supported but
not needed unless the alternative multi-label classification family is selected.

## D. Task decision record

| Field | Review |
|---|---|
| Decision / requirement | Support native primary L1/L2 identification and multi-label overlap localization without invented negatives or explanations |
| Alternatives | L1 sparse localization only; fixed-bin primary classification; maximal-primary-run classification; multi-label classification per bin; four complementary families; generic AnswerTask with typed label targets |
| Evidence | T4–T7; LabelSet.multi_label semantics; hello-world scoped ClassificationTask; Task stream in T6 |
| Hard constraints | Unknown labels excluded from primary supervision; no empty multi-label target interpreted as unknown; no synthetic QA/rationale; no ground-truth boundary leakage into a dense detection benchmark |
| Trade-offs | L1-only omits useful native L2/primary supervision. Fixed bins create many tasks but use a target-independent grid. Maximal-run scopes shrink task counts but reveal label-change boundaries and alter evaluation weighting. Multi-label classification is native but zero bins require a known-negative policy. Generic tensor answers preserve arrays but do not give ordinary classification semantics |
| Recommendation | **Judgment:** four families: fixed-bin primary L1 ClassificationTask, fixed-bin primary L2 ClassificationTask, sparse multi-hot L1 TemporalLocalizationTask, sparse multi-hot L2 TemporalLocalizationTask. Stream tasks rather than building a giant list |
| Justification | Addresses each available target natively; primary scopes follow the original fixed bin grid, not class-run transitions. Multi-hot per-class runs remain overlapping positive evidence, not fabricated negative annotations. Separate families permit explicit coverage and per-class evaluation |
| Verification | One primary task per valid hierarchy/bin, no -1 tasks, complete LabelSets, no skipped valid bins; two localization tasks/trace; independent class-run intervals and exact metadata; empty positive answer handling; family/split counts; reiterable stream with stable ordering/IDs and storage round trip; feature/prompt leakage audit |
| Confidence | Medium pending storage and throughput profiling. This is a recommended contract, not an experimentally established unique optimum |
| Open questions | Upstream task-volume limits and desired first-release families; benchmark before finalizing a public release. If task volume is unacceptable, propose a separately versioned compact profile rather than silently remove supervision |

**Observation:** training has 233,821 valid bins for each primary hierarchy,
so this profile creates **467,642 primary training tasks**, plus 2,248 localization
tasks. Primary constant-label run counts are only 14,287 / 14,649; that smaller
alternative is feasible but changes the question to operation recognition within
known homogeneous intervals. It is not an equivalent dense classification benchmark.
Do not use known-label run widths as inference inputs for full sequence detection.

**Fact:** set_task_stream supports a factory returning a fresh iterator; mixing
it with add_task/add_tasks is forbidden. Stream both task classes, register bounded
Records first, preserve factory repeatability and do not retain all Tasks in memory.
A streamed scope does not prove that a training adapter crops input tensors:
T7 returns full record series. Future adapters must explicitly implement the agreed
context policy. Task scopes identify an evaluation region; they do not authorize
copying annotations or target metadata into features. A bin index/timing query is
allowed only as target-independent task context, not derived from label values.

## E. Dataset identity, releases and partitions decision record

| Field | Review |
|---|---|
| Decision / requirement | Stable identity, exact source revision and official 1,124/160/160 trace partition |
| Alternatives | Source-matching williamhtan/kernelsight; authorized research namespace; new generated partitions (rejected) |
| Evidence | T8/T9/T10; pinned source card and split JSONs; TSQA generated-vs-official distinction |
| Hard constraints | No split regeneration, deduplication merging traces, or cross-split task references; IDs/card match; source ownership must not be misrepresented |
| Trade-offs | Source namespace is recognizable but technical discovery does not prove ownership authorization; research namespace is honest if clearly attributed but requires naming agreement. Dataset release, source commit, TimeF format and card schema are distinct versions |
| Recommendation | Propose williamhtan/kernelsight@1.0.0 with source revision pinned below, subject to maintainer namespace acceptance. Map train->TRAIN, val->VALIDATION, test->TEST on EVERY task, retain original provenance per Record |
| Justification | TimeF stores splits on Tasks; one trace remains one counter Record regardless of number of task families. Namespace can remain provisional during isolated fixture tests |
| Verification | Unique path union equals all 1,444 source inputs; pairwise disjoint partitions; native task splits match source owner; unique record counts correct; per-family task counts not confused with trace counts; read/write split round trip |
| Confidence | High on source/splits/APIs; Medium on namespace acceptance |
| Open questions | No technical ownership gate or blanket requirement for author permission was found in the inspected source. Maintainer approval is still needed for an upstream contribution; do not describe it as a license-imposed authorization requirement |

Never copy TSQA's StratifiedSplitter. Analysis tags are metadata views, not
partitions. Compiled release version 1.0.0, source release v4, source commit,
TimeF format 2, and YAML schema 1 must be recorded separately. Changed native task
families or timing policy require a new compiled release. Keep Python 3.9.6 Phase 1
environment intact; TimeNet requires a separate Python >=3.11 uv environment.

## Performance and acceptance gates

Analytical counter payload alone is 1,444*24*512*4 = 70,975,488 uncompressed bytes;
this is not an estimate of Parquet size. Stored original NPZ snapshot is
60,902,447 bytes including other files (see provenance report). Native task rows,
annotations and JSON can dominate control-plane size. No conversion benchmark
was run, and no fabricated timing/storage numbers are reported.

Before release record wall time, peak RSS, registry size, task scan throughput,
one-record read and representative batch read latency on stated hardware.
Use synthetic scale tests and training-only prototypes; retain the same hard
fidelity/leakage/split gates in every profile. Suggested engineering review
thresholds, not TimeNet limits: >2 GiB peak RSS or >1 GiB compiled size for this
small source triggers a design review. Streaming should avoid memory proportional
to total task count; test this directly. The resource thresholds are provisional
and require project acceptance; crossing them is not permission to drop data.

## Conclusion and status of earlier recommendations

Confirmed: scalar counter Signals/Parquet, exact edges plus native-us view,
unknown-vs-class distinction, immutable source pins, official task splits,
no synthetic explanations, separate environment and raw preservation.

Changed: L1-only localization is too narrow as the sole native interface;
metadata must be a lossless fallback alongside native supervision, not the only
access to L2/primary labels. Explicitly evaluate supported typed targets and
streamed tasks. Handle irregular final-bin scope correctly; payload metadata is
not automatically emitted by the standard torch adapter, and annotations ARE
emitted. Publication rights and namespace acceptance are different questions.

The source pin is resolved with full-file integrity evidence. Publication still
needs the attribution/notice review described in the provenance report. Task 2
can begin with local synthetic/training-only implementation and acceptance tests
under this proposed contract, but not as an approved public release. Confirm the
task profile and namespace before freezing a release. No upstream runtime tests
were run here; conclusions about APIs are static source findings, not a claimed
working connector.
