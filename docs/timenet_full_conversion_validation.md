# Phase 2 Task 3 — controlled full KernelSight conversion

For commands below, start at the Kernel X-ray repository root and set
`export KERNEL_XRAY_ROOT="$PWD"`. This publication-only path parameterization
does not change the recorded experiments or historical artifact contents.

## Scope and approved execution plan

This task mechanically converts the pinned KernelSight v4 source into a complete
**local** TimeF artifact. Validation/test arrays are decoded only for serialization
and integrity checks. They are not used for model, prompt, preprocessing or
conversion-design tuning. No model is trained and no dataset is published.

The source pin remains `d8cd8e804e5b619aadbb082a1fb3a8791b60a680`.
The development worktree is `.worktrees/timenet-kernelsight`, branch
`feature/kernelsight-v4`, based on TimeNet
`9bd1613019ee61fbc52e02106a1a45990ead7a54`. The optimized connector and two core
writer fixes match the Task 2.1 source hashes. Task 3 adds conversion/audit and
resource-monitor tooling, not a new storage schema or supervision contract.

The machine has **24 GiB physical RAM**. The initial memory-pressure query
reported 38% free, and disk had about 7.3 GiB free at first inspection (about
9.3 GiB immediately before conversion; free space is dynamic). The expected
compiled-size range was 450–650 MB based on training storage, with at least
4 GiB free required at launch and a **2 GiB free-disk stop reserve**.

A fresh training-only scaling job created 1,444 independently parsed Records:
all 1,124 training references plus 320 evenly spaced training duplicates with
unique synthetic identities. It changed no source files and decoded no held-out
NPZs. The 2 GiB guard stopped it during serialization at **2,147,516,416 bytes**
(2.00003 GiB), after completing conversion in 21.00 seconds. That synthetic
artifact is explicitly marked non-release and remains incomplete. Its task
counts are synthetic and are never substituted for real validation/test counts.

The user then **explicitly approved a 3 GiB RSS stop threshold**, keeping the
2 GiB free-disk reserve and separate writer/reader processes. The provisional
2 GiB target was therefore not silently relaxed. The 3 GiB threshold is a
controlled run limit, not a claim that the 2 GiB target is met or a model memory
budget. `outputs/phase2/task3/preflight.json` records the decision and evidence.

`ResourceGuard` samples own-process RSS, OS RSS high-water and free disk at a nominal
50 ms interval through macOS `proc_pidinfo`, `resource.getrusage` and `shutil.disk_usage`.
The OS high-water catches transient peaks missed between samples. A breach or
monitor failure writes diagnostics and exits nonzero; a final sample also checks
the limits. Sampling permits brief overshoot before termination; this is not a
kernel-enforced allocation quota. Observed maximum sample gaps were 0.823 s
(writer) and 0.104 s (reader), with OS high-water retained across those gaps.
Measurements apply to each isolated worker,
not all applications on the computer.

## Preserved representation and interruption safety

The approved representation remains:

- One Record and one fused-counter Source per trace, with 24 original scalar
  float32 counter Signals and 512 ordered values each, stored in Parquet.
- Canonical JSON array descriptors preserve every input/label/fingerprint key,
  dtype, shape and value, including primary -1, both multi-hot hierarchies, masks,
  segments, launch information, attributes, fingerprints, vocabularies and parents.
- All 513 original int64 nanosecond boundaries remain exact. Native TimeF uses
  relative microsecond floor coordinates and no invented wall-clock timestamp.
  Native coordinates are not represented as nanosecond-exact.
- The four task families remain valid-bin L1/L2 ClassificationTasks and per-trace
  sparse L1/L2 TemporalLocalizationTasks with all overlapping positive runs.
  Classification keeps the original fixed-bin evaluation units. Unknown primary
  labels do not become a class or negative supervision.
- Official JSON lists supply TRAIN/VALIDATION/TEST identity; no splitter or
  regenerated partition is used. `val` maps to TimeNet's `validation` enum value.
  Analysis tags do not define additional holdouts or prove identity independence.

Every worker refuses an existing output directory. The writer uses an isolated
`unvalidated-registry`, `replace=False`, UUID staging and the standard atomic
TimeF commit. An interrupted `.tmp-*` artifact has no accepted-release marker and
is never reused. A writer success is still marked **independent audit pending**.
Only a passing complete independent audit and preservation check permit a
Task 3 `COMPLETE.json` marker. Earlier registries, plots and reports are retained.
This local commit of storage files is not a git commit or remote publication.

## Independent integrity audit

The reader runs in a fresh process, reopens the written version with
`TimeFReader(open_local_version(...))`, and calls the official checksum verifier.
It first computes expected per-split/per-family counts directly from source
primary arrays and trace membership, without using the connector task factory.
It then visits every stored Record and compares reconstructed mappings to all
three original NPZs: complete key sets, shapes, dtypes, values and numeric bit
patterns. This also verifies exact ns edges and all auxiliary metadata.

For every native task the audit checks owner Record, official split, family,
class ID/text, full ordered vocabulary, target cardinality, original bin scope,
and absence of prompt, rationale or input annotations. It reconstructs primary
and multi-hot arrays, independently scans maximal positive runs, checks annotation
bin bounds plus exact ns metadata, and checks empty-positive answers. Record
boundaries include the final right edge. No paths or Record IDs may repeat or be
missing. Raw source hashes and official split records/indexes must match metadata.

Counter-only feature extraction must return only float32 [24,512] counter values,
bitwise equal to source measurements, through the existing whitelist. Synthetic
leakage tests remain applicable. Standard TimeNet torch items still expose
supervision: passing whole items/Records or metadata into a model remains unsafe.
This conversion does not implement an OpenTSLM model adapter.

Finally the audit runs TimeNet's relational control-database audit and compares
global task-table counts to both source expectations and the per-Record reader
counts. Record/Signal/task manifest counts must agree. Thus a task outside the
per-Record traversal cannot silently escape the audit. Source files are hashed
again, and a separate preservation inventory checks earlier research artifacts
and the original TimeNet checkout.

## Reproduction

Use the existing TimeNet uv/Python environment, without modifying the root
Phase 1 `.venv`. Use new output paths for every attempt. Before a new large run,
repeat resource preflight; the 3 GiB approval recorded here applies to this task.

```bash
export KERNEL_XRAY_ROOT="$PWD"
cd "${KERNEL_XRAY_ROOT}/.worktrees/timenet-kernelsight"
export UV_CACHE_DIR="${KERNEL_XRAY_ROOT}/.cache/uv"
export KERNELSIGHT_ROOT="${KERNEL_XRAY_ROOT}/data/kernelsight_v4"

.venv/bin/uv run python -m benchmarks.kernelsight.full_conversion \
  --mode write --data-root "$KERNELSIGHT_ROOT" \
  --output "${KERNEL_XRAY_ROOT}/outputs/phase2/task3/fresh-writer-output" \
  --rss-limit-bytes 3221225472

.venv/bin/uv run python -m benchmarks.kernelsight.full_conversion \
  --mode audit --data-root "$KERNELSIGHT_ROOT" \
  --version "${KERNEL_XRAY_ROOT}/outputs/phase2/task3/fresh-writer-output/unvalidated-registry/williamhtan/kernelsight/1.0.0" \
  --output "${KERNEL_XRAY_ROOT}/outputs/phase2/task3/fresh-reader-output" \
  --rss-limit-bytes 3221225472

.venv/bin/uv run pytest benchmarks/kernelsight/tests -q
```

The writer reports only factory counts and storage success; the independent audit
supplies actual source-versus-stored results. `resources.json` stores stage samples
outside the compiled version. Use `scale_training` with the same module invocation,
`--data-root`, a fresh `--output` and `--rss-limit-bytes 2147483648` to reproduce the
training-only scaling gate; its output must never be treated as the official data.

## Results

**Full conversion and integrity audit passed.** The complete artifact remains at
`outputs/phase2/task3/writer/unvalidated-registry/williamhtan/kernelsight/1.0.0/`.
Its build-time directory name is retained; Task 3's top-level `COMPLETE.json` and
passing audit identify its accepted status. It is not a public release.

| Measurement | Result |
| --- | ---: |
| Records | 1,444 |
| Counter Signals | 34,656 |
| Original NPZ mappings reconstructed | 4,332 |
| Native tasks | 601,292 |
| Bound positive-run annotation occurrences checked | 40,277 |
| Record conversion | 22.27 s |
| TimeF writing | 68.32 s |
| Writer process work, including factory traversal and preservation | 94.32 s |
| Writer peak RSS | 2,647,883,776 bytes (2.466 GiB) |
| Independent reader/audit process work | 58.86 s |
| Native task iteration and detailed checks within the audit | 27.78 s (~21,647 tasks/s) |
| Reader peak RSS | 1,118,420,992 bytes (1.042 GiB) |
| Compiled logical file size | 505,762,647 bytes (482.33 MiB) |
| Minimum observed free disk across writer/reader | 8.79 GiB |

The writer **exceeds the original 2 GiB criterion** and passes the explicitly
approved 3 GiB run limit, with about 0.534 GiB headroom. Neither process crossed
the disk reserve. Reported runtimes are individual local observations, not
statistical performance guarantees. Process work excludes interpreter/import and
initial source-inventory setup. Arrow logged restricted CPU-capability sysctl
warnings during the reader run; checks still passed, and timings reflect that
execution environment. No performance-driven tuning used held-out contents.

Actual independently source-counted and reader-verified task counts:

| Official partition | Records | Primary L1 | Primary L2 | L1 localization | L2 localization | Total tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,124 | 233,821 | 233,821 | 1,124 | 1,124 | 469,890 |
| val | 160 | 33,416 | 33,416 | 160 | 160 | 67,152 |
| test | 160 | 31,965 | 31,965 | 160 | 160 | 64,250 |
| Total | 1,444 | 299,202 | 299,202 | 1,444 | 1,444 | 601,292 |

These are measured mechanical counts, not extrapolations. All source arrays,
float bit patterns, exact ns edges, task targets, annotation intervals and split
metadata passed. Full-record and global-catalog task counts agree with independent
source expectations; no missing, duplicated or cross-partition Records were found.
Source fidelity refers to arrays, not original ZIP compression bytes.

The entire 4,345-file pinned source inventory is verified: 4,341 files under the
raw root plus three verbatim documents under `docs/dataset/` and the preserved
root `.gitattributes`. `source_hashes.json` stores every expected SHA256.
Preservation checks cover 21,931 existing workspace files and 497 original
TimeNet source files, compatibility aliases and original git status. The raw
dataset, splits, Phase 1 code/environment/galleries, reports and all earlier
benchmark artifacts remain unchanged. Recheck with:

```bash
: "${KERNEL_XRAY_ROOT:?Run the repository-root setup block above first}"
cd "${KERNEL_XRAY_ROOT}"
.venv/bin/python scripts/verify_task3_preservation.py
```

## Evidence and remaining requirements

Machine-readable deliverables are
[`full_conversion_report.json`](../outputs/phase2/task3/full_conversion_report.json)
and [`full_integrity_audit.json`](../outputs/phase2/task3/full_integrity_audit.json).
The directory also contains preflight evidence, the failed synthetic scaling
run, separate writer/reader `result.json` and resource samples, source/compiled
hashes, exact implementation hashes, environment versions and test logs.

The environment is Python 3.13.4, TimeNet/connectors 0.12.0, NumPy 2.5.0, Arrow
24.0.0, DuckDB 1.5.6 and Pydantic 2.13.4 on arm64 macOS. Original Phase 1
Python 3.9.6 remains separate and unchanged. No new dependency was installed.

The six new synthetic tests exercise the full independent audit over all three
partition identities (including empty answers), missing Records, a deliberately
cross-partition reference, independent overlapping runs, emergency RSS exit and
the final resource check. Connector tests retain leakage sentinels and real
training-only integration tests. Final complete check results are recorded in the
conversion report.

| Final check | Result |
| --- | --- |
| Full TimeNet `make test` | 2,896 passed, 2 skipped, 5 warnings; 260.26 s |
| Synthetic audit + connector + memory-regression tests | 29 passed, no skips; 2.74 s |
| `make test-connectors` | All environments/types passed; KernelSight default 15 passed, 2 opt-in skips |
| `make check`, `make lint-fix`, `make check-ci` | Passed, including minimum Python 3.11 typing |
| Explicit hooks over all 25 new/modified files | Passed |
| Phase 1 loader tests | 32 passed; 11.91 s |
| Existing visualizations | Verification passed; no plots regenerated |
| Final preservation inventory | Passed; 21,931 workspace + 497 upstream source files unchanged |

The two skipped core cases and five fork/pytest warnings remain visible in the
full test log. The 29-test targeted run includes the opt-in real **training-only**
connector tests; full held-out verification is performed by the mechanical audit,
not by widening development-test access.

The implementation is technically suitable for an upstream maintainer review
under the measured resource requirements. Include the two small core writer
fixes and their regression evidence; maintainers still need to accept the task
profile, metadata volume and connector namespace. This is **not** publication or
release approval: namespace acceptance, collection-version/third-party notice
completeness and actual license/notice delivery with redistributed artifacts
remain unresolved. The card's Apache-2.0 declaration alone does not close those
requirements. No commit, push, PR, publication, new labels, explanations or model
training was performed.
