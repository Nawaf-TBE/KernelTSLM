# Kernel X-ray research instructions

## Objective and current scope

Investigate whether OpenTSLM can (1) identify GPU operations over time,
(2) detect overlapping execution, and (3) explain results grounded in hardware
counter evidence. This phase prepares and visualizes data. Do not train models
without a subsequent explicit request. Do not assume OpenTSLM architecture,
input requirements, or performance before checking the chosen implementation.

## Dataset and preservation

KernelSight v4 is already downloaded to the project root. Preserve `kernels/`,
`splits/`, and upstream `tools/` unchanged. Do not regenerate, rename, normalize,
or overwrite raw NPZ files or official split records. Save derived artifacts
under `outputs/` and research reports under `reports/`. Upstream documentation
was relocated verbatim to `docs/dataset/`; its paths refer to the dataset root.

Each snapshot contains sibling `input/tensor_input.npz`, `labels/labels.npz`,
and `fingerprint/fingerprint.npz`. Split record `path` points to the input and
is relative to the project root. Some motifs have no variant subdirectory.

- Input `data`: float32 [24, 512]; `counter_names`: 24 names;
  `time_edges_ns`: 513 timestamp edges. Use actual edges; bin duration varies
  across traces. Counters include pipe activity, memory, warp-stall,
  system-throughput, and SASS groups, already physically scaled upstream.
- Primary targets: `workload_l1` and `workload_l2`, each [512]; -1 is unlabeled.
  `mask_labeled` marks primary-label supervision. `mask_any_kernel` describes
  kernel activity and is not interchangeable with the supervision mask.
- Multi-hot targets: `workload_l1_multihot` [512, 12] and
  `workload_l2_multihot` [512, 73]. Derive multi-class overlap from at least two
  active L1 entries; check `multihot_has_overlap` against this definition.
- Read vocabulary order from each NPZ: `vocab_l1`, `vocab_l2`, `l2_parent_l1`.
  Segment arrays, attribute flags, and fingerprint vectors are additional data.
- Object metadata requires `allow_pickle=True`; load only trusted local files.

## Completed work

All 1,444 traces downloaded and validated with zero reported issues. Original
report: `reports/dataset_validation.json` (moved from the root, contents intact).
Tensor and label schemas inspected. Official split counts verified:
train 1,124, validation 160, test 160; no shared trace paths across these splits.
299,202 / 739,328 bins are labeled (40.47%); 472 traces are overlap-flagged;
61,974 bins have multiple active L1 classes. There are 11 observed primary L1
classes out of 12 vocabulary entries, and substantial matmul imbalance.
Phase 1 Task 6 is implemented by `scripts/visualize_traces.py`; inspect its
manifest and `reports/phase1_visualization_verification.json` for verification.

## Evaluation constraints

- Develop plots, prompts, heuristics, and examples using training only.
  Do not inspect test traces for development. Validation is for subsequent
  model selection; test is reserved for final evaluation.
- Official train/val/test form a disjoint trace partition. `iid`, `param_ood`,
  `composed`, and `length_ood` are overlapping analysis tags, not independent
  holdouts. Do not infer kernel-identity independence from path disjointness.
- Never feed labels, multi-hot arrays, segments, filenames, split metadata, or
  fingerprints containing target information into a counter-only model.
  Record any change to the input modality as an explicit experiment.
- Mask unlabeled bins for primary classification. Do not interpret -1 as a
  negative example or assign it a workload class. Plan macro/per-class metrics
  alongside aggregate scores because of imbalance. State evaluation coverage.
- Evaluate multi-label overlap separately from primary operation recognition;
  same-class concurrent kernels may not register as multi-class overlap.
- Overlap labels describe device-marker envelopes at launch granularity; they
  do not prove instruction-level simultaneity. Counter evidence does not
  establish causation. Explanations must distinguish observations from
  hypotheses and acknowledge coarse or replay-derived measurements.
- Fit future preprocessing only on training. Plot p99 normalization is
  per-trace display scaling, not a proposed training transform or a way to
  compare absolute activity across channels/traces.

## Running and changing code

Use `python scripts/visualize_traces.py`; see `docs/visualization.md`.
Keep deterministic selection, record source hashes and versions, verify time
alignment and label consistency, and visually inspect outputs after plot edits.
Preserve old exploratory plots. Do not invent absent collector scripts or claim
the upstream SASS loader is a ready OpenTSLM adapter.

## Local visualization environment

Apple `/usr/bin/python3` is Python 3.9.6. Use `.venv/bin/python` for project
commands; Homebrew is not installed or required. Dependencies are pinned in
`requirements-visualization.lock.txt` and included by `requirements.txt`.
Setup and rendering were checked under Python 3.9.6; the compatible gallery
is in `outputs/phase1/python39/`, with results in
`reports/python39_visualization_verification.json`.

## Phase 1 Task 7: model-ready records

`kernel_xray/dataset.py` provides `KernelSightDataset`, `collate_records`, and
explicit training-only `fit_training_statistics`. The loader supports official
train/val/test indexes; analysis tags/custom files are rejected. Preserve NPZ
channel/time order, raw float32 counters, absolute int64 edges, primary -1 labels,
and both multi-hot hierarchies. Only `features` is counter-only model input;
metadata, targets, and label-derived masks must not leak into model features.

Optional population z-score statistics use all official training bins, including
unlabeled/inactive bins, accumulated in float64. They are never fitted implicitly
or on held-out data. Raw measurements remain a separate tensor. Statistics JSON
records training input/split hashes and channel order. See `docs/model_records.md`
for masks, zero-variance handling, and overlap supervision caveats.

Run `.venv/bin/python scripts/verify_model_records.py` to reproduce tests,
training-only raw/normalized batch checks, and preservation checks. Reports go to
`reports/task7_verification.json` and `reports/task7_tests.xml`; statistics go to
`outputs/phase1/task7/train_statistics.json`. Dependency pins are in
`requirements-records.lock.txt`, which includes the unchanged visualization lock.
Tests enforce no real validation/test NPZ reads; synthetic held-out fixtures test
those API branches. Existing visualization outputs must remain untouched.

Task 7 verification completed on Python 3.9.6 / PyTorch 2.6.0: 32 tests
passed with no skips. All 1,124 training records were compared to source NPZs;
normalization covers 575,488 training bins. Original raw/split/tool/output hashes
were unchanged. No models were trained and no real held-out NPZs were opened.

## Repository layout migration

Source code now lives in `src/kernel_xray/`. Raw data and upstream tools are
physically under `data/kernelsight_v4/`; root `kernels`, `splits`, and `tools`
are compatibility symlinks. Preserve both the targets and aliases. Reports live
in `outputs/reports/` with a root `reports` compatibility symlink. Existing
`outputs/phase1/` galleries/statistics and `outputs/plots/` are retained unchanged.
The earlier `kernel_xray/dataset.py` references mean `src/kernel_xray/dataset.py`.
Use scripts as before; `scripts/validate_dataset.py` runs Task 7 verification.
No real held-out tensor inspection or model training is authorized by migration.

## Phase 2 Task 2 local connector

The unsubmitted TimeNet connector is in `.worktrees/timenet-kernelsight` on
`feature/kernelsight-v4`, based on `9bd1613019ee61fbc52e02106a1a45990ead7a54`.
It has its own uv/Python 3.13 environment; do not replace the Phase 1 `.venv`.
Namespace and publication approval remain pending. Real development and scale
runs are training-only; no full validation/test conversion is authorized here.
See `docs/timenet_connector_task2.md` and the connector README for commands,
acceptance outcomes, and the peak-memory review gate. Preserve the existing
Phase 1 outputs and the verified source revision.

## Phase 2 Task 2.1 memory optimization

See `docs/timenet_memory_optimization.md` and `outputs/phase2/task2_1/acceptance.json`.
The complete training-only benchmark now peaks at 1.954 GiB (profiled repeat
1.956 GiB), retaining all 469,890 tasks and exact reconstruction of 3,372 NPZs.
The worktree includes two minimal core writer fixes: 8 MiB hierarchy payload
batches and Arrow relation insertion avoiding transactional catalog retention.
Connector allocation reductions preserve independent mutable metadata containers.
The original benchmark/code and Phase 1 environment/output hashes are preserved.
Full conversion remains a separately authorized task; the small training headroom
does not establish that 1,444 traces fit 2 GiB. Do not publish or train.

## Phase 2 Task 3 controlled full conversion

The user explicitly authorized held-out NPZ decoding solely for mechanical full
conversion and integrity, and approved a 3 GiB RSS stop after training-only
synthetic scaling exceeded 2 GiB. This does not authorize held-out development,
model/prompt tuning, training, or publication. The complete local artifact is under
`outputs/phase2/task3/writer/unvalidated-registry/`; consult Task 3's top-level
`COMPLETE.json`, conversion report and independent integrity audit before use.
The directory name is deliberately retained from the pre-audit build.
See `docs/timenet_full_conversion_validation.md` for measured resources and exact
per-partition counts. Recheck preservation with the Phase 1 environment:
`.venv/bin/python scripts/verify_task3_preservation.py`. Namespace approval,
collection-version notices and redistribution requirements remain unresolved.
