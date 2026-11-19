# Proposed KernelSight–TimeNet implementation specification

18 November 2026. This specification contains supported API choices, not a working
connector or an authorization to publish. The [decision review](timenet_decision_review.md)
explains alternatives; the [provenance report](source_provenance_and_license.md)
records verified source and unresolved publication evidence. These documents
supersede differing recommendations in the earlier integration plan.

## 1. Supported implementation boundary

- TimeNet checkout: `9bd1613019ee61fbc52e02106a1a45990ead7a54`.
- Upstream HF dataset: `williamhtan/kernelsight`.
- Verified SOURCE_REVISION: `d8cd8e804e5b619aadbb082a1fb3a8791b60a680`.
- Source release: KernelSight v4. Compiled release proposed: 1.0.0.
- Proposed ID: `williamhtan/kernelsight`, pending namespace acceptance. Do not
  imply dataset-author participation or approval merely by using that ID.
- TimeF format 2, YAML card schema 1; separate Python >=3.11 uv environment.
  Do not install TimeNet into or replace Kernel X-ray's Python 3.9.6 `.venv`.
- No model, synthesized supervision, normalization, split generation or publishing.

## 2. Connector package and acquisition

Future directory, in an authorized development checkout:

```text
packages/timenet-connectors/src/timenet_connectors/datasets/
  williamhtan/__init__.py
  williamhtan/kernelsight/
    __init__.py          # exports CONNECTOR
    connector.py         # BaseConnector, no-argument construction
    dataset.yaml
    requirements.txt
    source.py            # pinned acquisition and strict NPZ parsing
    mapping.py           # canonical encode/decode, counter Signals, exact indexes
    tasks.py             # native task stream factories
    README.md            # contract, licenses, provenance and coverage
    tests/
      test_source.py
      test_mapping.py
      test_tasks.py
      test_roundtrip.py
      fixtures/          # independently synthetic data only
```

Implement BaseConnector with download/download_async and offline convert.
Use immutable hub_snapshot acquisition or a read-only proposed KERNELSIGHT_ROOT
local mode; that name is connector-owned configuration, not an existing TimeNet
setting. Declare dependencies independently, lazy-import downloads, verify pinned
file sizes/hashes, and never use the user's raw directory as disposable cache.
No PyTorch dependency is needed merely to parse NPZs. Retain raw cache when needed
and materialize arrays before source cleanup, or explicitly maintain lazy lifetimes.
Never modify existing upstream tools. Configuration/source-mismatch failures must
use TimeNet exceptions, not silent repair.

**Acceptance S1:** discovery.resolve finds CONNECTOR and matching card; no-argument
construction; convert runs offline on synthetic refs; missing/corrupt/pin-mismatched
sources fail; local source hashes are unchanged after both successful and failed
builds. Full upstream snapshot inventory matches the saved 4,345-file audit.

## 3. Trace and timing model

Construct TimeFDataset(metadata=connector.metadata()). For each original trace,
create one Record, one Source for the fused counter bundle, and 24 separately owned
Signals. Use TimeSeriesSpec with scalar float32, dimensionless units, and original
Signal names plus explicit channel index. Signal.from_irregular accepts the 512
counter values and 512 int64 relative microsecond left offsets. Use original JSON
order; do not deduplicate traces or reorder channels. A complete source build has
1,444 counter Records and 34,656 counter Signals. Additional label Signals are
not part of the selected input representation.

Store e[0:513] exactly as int64 nanosecond values in canonical metadata. Derive
q[i]=(e[i]-e[0])//1000 using overflow-safe integer arithmetic, not floating
seconds. TimeOrigin.timestamp remains None. Record.time_span covers [q[0],q[512]).
For native task scopes use whole-record TimeIntervals; explicitly scoped Signal
windows require care because an irregular Signal ends at its last sample+1 us.
Reject collapsed native bounds or invalid extents. Do not stretch intervals.

**Acceptance S2:** bitwise float32 values and channel order round-trip through
Parquet and inverse stacking; no hidden transforms; exact original 513 edges and
origin recover; native q/error rule matches every boundary; final-bin tasks pass
TimeF validation; synthetic sub-us bins fail clearly. Training measured max error
999 ns and minimum q-bin width 200 us are checks on observed data, not a guarantee
about uninspected held-out arrays.

## 4. Canonical supervision and auxiliary payload

Use a versioned namespaced JSON-native payload (schema_version 1) with original
array dtype/shape/value descriptors. Preserve signed int32 primary arrays, uint8
multi-hot arrays and original masks, counts/flags, segment_id, all source segment
arrays/attributes/names, input launch arrays/function IDs, full ordered vocabularies,
L2 parent mapping, spatial-state vocabulary, and every fingerprint key. Preserve
source path, split record, official list index and source hashes separately.
Do not decode arbitrary untrusted object pickles; expected text metadata is loaded
only from the verified local/pinned dataset. Unknown keys/types fail rather than
vanish. Native representations below supplement, not replace, this exact payload.

**Acceptance S3:** inverse decoder recovers every source key, shape, dtype, integer
and numeric array value; -1 remains -1; source masks remain distinct; all 12/73
entries exist in original order even when unobserved; no fabricated spatial-state
track; fingerprints are not assumed to have shape [32]. Compression/container
byte identity is not promised, but original source SHA-256 is retained.

## 5. Recommended native task contract

This contract is API-supported and recommended; finalize the public release
profile after maintainer feedback and a training-only scale benchmark.

| Family metadata tag | TimeF class | Targets and scope |
|---|---|---|
| primary_l1 | ClassificationTask | One textual class for each valid original L1 bin; fixed original-bin TimeInterval scope |
| primary_l2 | ClassificationTask | Same for L2, with its separate full LabelSet |
| multihot_l1_localization | TemporalLocalizationTask, SPARSE | Per-class contiguous positive runs, attached bound annotation occurrences |
| multihot_l2_localization | TemporalLocalizationTask, SPARSE | Same for every active L2 class; overlapping runs retained |

Classification label sets list all vocab entries as Label(name=original_name),
in original order. Do not insert unknown into them. Skip primary -1 bins;
skipping is recorded as supervision coverage, not a new class. Class IDs and
exact bin indexes remain in task metadata. Fixed-bin scopes are independent of
class changes; maximal homogeneous runs are not equivalent evaluation units.
Do not generate empty multi-label classification negatives from all-zero tracks.

Localization annotations carry class text, ID, exact bin start/stop and source ns
bounds. Attach them to Records, retain returned occurrences, and reference these
in target_annotations. Do not also set inline targets. Empty positive answers use
explicit targets=() with no target_annotations and must round-trip. Original
segments are preserved separately, not regenerated from these runs. Keep
rationale absent and no explanation/QA target generation. Query text, if used,
is a fixed neutral task instruction, never a generated factual explanation.

Register Records before tasks. Use the actual
TimeFDataset.set_task_stream((ClassificationTask, TemporalLocalizationTask), factory)
API with a fresh deterministic iterator on each invocation; do not mix add_task
with task streaming. Stable family/record/bin ordering is part of the contract.
Training-only expected primary-task counts are 233,821 per hierarchy, plus 1,124
localization tasks per hierarchy. Validate final per-split counts mechanically;
do not infer held-out content statistics for design.

**Acceptance S4:** complete, nonduplicated valid primary-bin coverage with no -1
class; label sets and target classes agree; reconstruct both multi-hot matrices
from annotation indexes, preserving overlaps and gaps; source masks/counts agree;
empty answers validate; reiterating the stream yields identical semantics; actual
writer/reader preserves task family, scope, target occurrences and split.

## 6. Partitions and input leakage

Map original train/val/test JSONs to Split.TRAIN/VALIDATION/TEST on every task.
No splitter is called. Record metadata carries original partition and index because
Record has no native split field. Unique counter Records per partition remain
1,124/160/160, regardless of task counts. Analysis tags remain overlapping metadata.

The only model features are the 24 counter arrays. TimeFTorchDataset._item emits
all input Record Signals AND annotations/tasks; it does not automatically expose
Record.metadata or crop tensors to task.scope. A model adapter must construct a
whitelist-only feature dictionary, handle supervision separately and enforce the
agreed full-context/bin-scope policy. Empty input_annotations alone is not a
leakage guarantee. The adapter must never stringify a whole Record into a prompt.

**Acceptance S5:** official path/hash union and disjointness; no task points to an
input Record from another split; all four families preserve owner splits; split
accessors survive storage round trip. Test the feature-extraction boundary with
sentinel target strings/classes/paths so none appears in feature tensors/prompts;
permuting target metadata cannot change input arrays. No label-bearing Signal
may be attached to input Records. Standard torch masks are not supervision masks.

## 7. Reproducibility, test and release gates

Use synthetic fixtures for edge cases and real training only for development.
A separately controlled full-byte conversion may touch held-out tensors solely
for serialization and integrity, never tuning. This review hashed held-out bytes
without decoding them. A future full conversion is distinct from a model test.

**Acceptance S6:** LocalRegistry.put_dataset -> TimeFReader(open_local_version(...))
round trip verifies S1–S5 with raw/derived boundaries distinguished. Repeated
conversion has identical canonical values/task semantics. Capture code commit,
source revision, input/split hashes, dependency/build environment, registry size,
peak RSS, conversion and read costs. The native task profile is provisional until
scale tests establish resource feasibility; review thresholds in the decision
report are proposed project criteria, not TimeNet capabilities or measured limits.

**Acceptance S7:** TimeNet make check, make lint-fix, make test,
make test-connectors, make license-check; CI parity make check-ci/make test-unit.
Keep its Python >=3.11 environment separate. Recheck Kernel X-ray's existing 32
loader tests and visualization verification without regenerating completed plots.
Hashes of raw/split/upstream-tool/Phase 1 output files must remain unchanged.
No runtime API or conversion tests were executed in this documentation task.

**Acceptance S8 (publication only):** close namespace acceptance, complete
collection-version/third-party notice evidence, and confirm actual delivery of
license texts/change notices with compiled artifacts. Card Apache-2.0 metadata
and a citation alone are not that delivery. Source integrity is resolved;
redistribution notice completeness is not. Do not publish until S8 is met.

## 8. Task 2 start order and outstanding decisions

1. Confirm the proposed four-family contract and native-us/exact-ns distinction.
2. Create only a local authorized implementation branch/worktree/environment;
   add synthetic parser, mapping, inverse-decoder and task tests before full data.
3. Prototype the selected scalar/Parquet design; prove scope and end-bin behavior.
4. Stream native tasks and measure training-only scale costs. Seek maintainer
   feedback on canonical metadata volume and task profile before freezing release.
5. Run round trips, leakage tests and separate upstream/project regression checks.
6. Authorize and execute controlled full conversion/integrity, preserving partitions.
7. Resolve publication evidence and namespace before any public release proposal.

Unsupported claims deliberately excluded: exact nanosecond-native TimeF support,
a masked dense-classification task class, metadata isolation by default, automatic
scope cropping in torch, a ready OpenTSLM adapter, instruction-level overlap truth,
complete third-party notice verification, or maintainer approval already obtained.

Task 2 **may begin as local synthetic/training-only implementation** once the
proposed profile is accepted. It is not yet cleared for publication, model training,
or release-finality claims. An ns-format extension is necessary only if native ns
semantics become a requirement, not for exact source preservation in this design.
