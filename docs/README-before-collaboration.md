# Kernel X-ray

Research into whether OpenTSLM can identify GPU operations, detect overlapping
execution, and explain predictions using GPU hardware counters. The current
milestone is **Phase 1 Task 7: model-ready records**, following verified trace visualizations. No models have
been trained in this workspace.

## Project structure

```text
Kernel X-ray/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── data/kernelsight_v4/       # unchanged kernels/, splits/, upstream tools/
├── src/kernel_xray/
│   ├── dataset.py
│   ├── preprocessing.py      # public training-statistics API
│   ├── visualization.py
│   └── evaluation.py         # split-isolation utilities
├── scripts/                  # validation, visualization, batch examples
├── tests/
├── configs/
├── experiments/
├── docs/
└── outputs/
    ├── plots/                # retained exploratory plots
    ├── phase1/               # retained verified galleries and statistics
    └── reports/
```

The dataset stays at its original paths: split entries resolve relative to this
project root. Upstream documents are preserved verbatim under `docs/dataset/`;
paths in their examples still refer to the dataset root. Some upstream schema
notes describe collector artifacts absent from this distribution; use the v4
dataset card, actual NPZ fields, and official split JSONs for this release.

## Run Phase 1 visualizations

The visualization tools support Python 3.9–3.12, including this Mac's
`/usr/bin/python3` (3.9.6). Homebrew is not needed. From the project root:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-visualization.lock.txt
.venv/bin/python scripts/visualize_traces.py
.venv/bin/python scripts/verify_visualizations.py
```

These commands use the project environment explicitly, so shell activation and a
system `python` command are unnecessary. To activate it optionally, run
`source .venv/bin/activate`; then `python` points to the project environment.
`requirements.txt` includes the same pinned dependencies.

Plots and a provenance manifest are written to `outputs/phase1/traces/`.
See [visualization instructions](docs/visualization.md) for custom paths,
stored-value views, and plot interpretation. Existing plots record the versions
that actually rendered them; a different environment may change PNG bytes.

## Model-ready records (Task 7)

Use the existing `.venv`:

```bash
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python scripts/verify_model_records.py
```

This runs automated tests, fits training-only normalization statistics, checks
raw/normalized batches, and records results in `reports/task7_verification.json`.
The loader supports official train/val/test indexes, preserves raw counters and
timestamps, and returns primary/multi-hot targets, separate masks, and metadata.
See [model-record instructions](docs/model_records.md) for the API, sample shapes,
normalization decisions, and held-out evaluation boundaries. No model is trained.

## Dataset status and evaluation

KernelSight v4 contains 1,444 traces, each with 24 counter channels and 512 bins.
The existing [validation report](reports/dataset_validation.json) records zero
issues, 299,202 labeled bins (40.47%), 472 overlap-flagged traces, and 61,974
multi-class bins. There are 12 L1 vocabulary entries, 11 observed primary L1
classes, and 73 L2 vocabulary entries. Matmul dominates.

Official train / validation / test counts are 1,124 / 160 / 160, with zero shared
trace paths. Develop plots and examples using **training only**. Reserve
validation for later model selection and test for final evaluation. Labels are
ground truth, never model inputs; unlabeled bins require masking. See
[AGENTS.md](AGENTS.md) for the full research and evaluation constraints.

## Layout compatibility

Raw dataset directories now live under `data/kernelsight_v4/`. Root-level
`kernels`, `splits`, and `tools` are compatibility symlinks to those directories;
`reports` links to `outputs/reports`. Dataset bytes and split-relative paths are
unchanged. Existing APIs default to the project root through these aliases;
explicit `--data-root data/kernelsight_v4` also works for visualization.

Use `.venv/bin/python scripts/validate_dataset.py` for training-record verification
and `.venv/bin/python scripts/visualize_traces.py` for plots. The validation command
checks split indexes but does not open real held-out tensors. Source modules now
live under `src/`; scripts and pytest configure this import path. To import the
package from other applications, install it with `.venv/bin/python -m pip install -e .`.
The preprocessing module exposes the existing training-statistics implementation;
the evaluation module checks split isolation and does not evaluate a model.

## TimeNet connector — Phase 2 Task 2

The unsubmitted implementation is in `.worktrees/timenet-kernelsight` on
`feature/kernelsight-v4`, with a separate uv/Python 3.13 environment.
[Commands, verification results, and scalability limits](docs/timenet_connector_task2.md)
are documented. Synthetic and real-training storage tests pass, and all 1,124
training traces and 469,890 native tasks were reconstructed through TimeF.
Task 2 initially exceeded the provisional 2 GiB criterion.
[Task 2.1 memory profiling and optimization](docs/timenet_memory_optimization.md)
reduced the complete training benchmark to 1.954 GiB with full correctness retained.
Those measurements covered training only; Task 3 results below cover the full
dataset.
No real held-out conversion, training, publication, commit, or PR was performed.

## Complete local TimeF conversion — Phase 2 Task 3

[Full conversion and independent integrity audit](docs/timenet_full_conversion_validation.md)
cover all 1,444 traces, 34,656 counter Signals and 601,292 native tasks, preserving
the official 1,124 / 160 / 160 partitions and all 4,332 source NPZ mappings.
The writer peaked at 2.466 GiB and the separate reader at 1.042 GiB, within the
explicitly approved 3 GiB stop threshold. The original 2 GiB criterion is exceeded.
Use `outputs/phase2/task3/COMPLETE.json` and the linked reports to identify the
accepted artifact. This is a local research conversion, not a published dataset
or authorization to use held-out data for development. Licensing/namespace
requirements remain unresolved; no model has been trained.
