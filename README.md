# Kernel X-ray

### Explaining GPU Execution from Hardware Time Series

Can hardware-counter time series identify GPU operations, resolve overlapping
execution, and support explanations grounded in measured evidence?
Kernel X-ray investigates this question with KernelSight v4 and a verified
counter-only CNN baseline.

**[Read the CNN research results →](docs/cnn-baseline.md)**

## Research context

This TimeNet hackathon research project represents GPU traces in TimeF and builds
reproducible recognition baselines before studying language-model explanations.
TimeNet provides dataset representation and task infrastructure; OpenTSLM is a
planned research direction, not an implemented or validated capability here.

## Dataset

KernelSight v4 contains **1,444 traces**, **24 counter channels**, and **512 bins**
per trace, with 513 original timestamp edges, **12 L1** and **73 L2** vocabulary entries.
The official partition is **1,124 train / 160 validation / 160 test**.
Unknown primary labels remain -1 and are masked. Channel order and original
nanosecond timestamps are preserved. Test data is reserved for final evaluation.

## Research goals

- Recognize GPU operations over time.
- Evaluate overlapping execution separately from primary classification.
- Study temporal reasoning and transition behavior.
- Generate explanations grounded in hardware-counter observations.

Counter evidence does not establish causality. Marker-envelope overlap does not
prove instruction-level simultaneity.

## Completed work

1. Phase 1: validated all traces, built verified training-only visualizations,
   and implemented a reproducible PyTorch loader (32 tests at that milestone).
2. Phase 2: implemented a local TimeNet connector, profiled memory, and verified
   full mechanical conversion while preserving official partitions.
3. Phase 3: prepared physically separate train/validation TimeF views, trained
   the L1 CNN, froze its selected checkpoint, and independently audited results.

## CNN validation results

| Experiment | Macro-F1, fixed 11 | Accuracy | Best epoch |
|---|---:|---:|---:|
| Unweighted, frozen | 0.945925 | 95.93% | 7 |
| Weighted | 0.943411 | 95.88% | 13 |

**VERIFIED WITH QUALIFICATIONS.** CPU replay matches every saved prediction.
These are official-validation results on 33,416 annotated primary L1 bins,
covering 40.79% of validation positions. All validation collector families occur
in training; unseen-workload claims remain preliminary. Boundary accuracy is
63.92%. The macro-F1 trace-bootstrap interval is approximately [0.831, 0.969].
Single-seed MPS nondeterminism and workload dependence qualify reproducibility.

See the [results page](docs/cnn-baseline.md),
[full independent audit](docs/phase3_cnn_independent_audit.md), and
[lightweight result provenance](results/cnn/provenance.json).

## Repository structure

```text
AGENTS.md                    Research and preservation constraints
README.md                    Project entry point
CONTRIBUTING.md              Collaboration workflow
pyproject.toml               Phase 1 package and dependency configuration
src/kernel_xray/             Loaders, visualization, evaluation, Phase 3 CNN
scripts/                     Reproducible verification and experiment tools
tests/                       Synthetic and guarded real-data checks
configs/                     Experiment configurations
experiments/                 Experiment workspace
docs/                       Methods, audits, and results documentation
results/cnn/                 Lightweight presentation assets and provenance
data/kernelsight_v4/         LOCAL raw data and official splits
outputs/                     LOCAL derived artifacts and checkpoints
.worktrees/timenet-kernelsight/ LOCAL separate TimeNet Git worktree
```

Root kernels/splits/tools and reports aliases preserve historical local paths.
They are excluded from publication along with their local targets.

## Setup and reproduction

Phase 1 uses the existing Python 3.9-compatible `.venv` and pinned requirements.
In a prepared workspace, verified checks are:

```sh
.venv/bin/python scripts/verify_model_records.py
.venv/bin/python scripts/verify_visualizations.py
```

Phase 3 uses a separate TimeNet Python 3.13 environment. Do not mix its
requirements with the root Phase 1 pins. Read the
[pipeline guide](docs/phase3_baseline_data_pipeline.md) and
[CNN reproduction guide](docs/cnn-baseline.md#reproducibility-and-artifact-access).
A clean clone does not include datasets, views, environments, or checkpoints.
Authorized collaborators must arrange private access with the owner and verify
upstream permissions. No download link or redistribution license is assumed.

## Collaboration

Use feature branches and reviewed pull requests; keep main stable.
Document seeds, split membership, normalization, metrics, and limitations for
experiments. Do not use the official test set for iterative development.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

The TimeNet worktree is an independent repository and must never be flattened
into this repository or have its history rewritten by project publishing.

## Publication scope

Source, project-authored documentation, tests, configurations, and selected small
results are candidates for tracking. Raw data, compiled releases, checkpoints,
credentials, and full outputs remain local. See [publication manifest](docs/publication_manifest.md).
No teammate is added until their GitHub username is provided.

## Planned research

Linear baselines, temporal-order ablations, L2 recognition, overlap detection,
OpenTSLM integration and grounded explanations remain planned; conventional QA
reference baselines are complete below.
## QA Benchmark and Baselines

QA v1 contains **19,021 verified questions**: interval means, counter-maximum
comparison, and unique-peak ordering. A deterministic oracle and independent
scalar validation verify every answer; exact regeneration passed.

| Validation method | Comparison | Temporal |
|---|---:|---:|
| Majority | 51.125% | 37.581% |
| Counter-pair lookup | 56.125% | 61.769% |
| Primary logistic regression | 93.375% | 62.289% |
| Calculation-aware LR | 98.125% | 89.597% |
| Direct calculation control | 100% | 100% |

**VERIFIED WITH QUALIFICATIONS.** Primary temporal LR does not establish useful
reasoning beyond the shortcut (+0.52 points, paired 95% CI [-2.12,+3.08]).
Calculation-aware features explicitly expose answer-sufficient quantities; direct
calculation is an arithmetic control. Structured parameters, normalized counters,
workload overlap and identity shortcuts limit interpretation. No OpenTSLM
training/evaluation or official QA test evaluation is claimed.

See [QA definitions](docs/phase4_qa_v1.md), [baseline methods](docs/phase4_qa_baselines.md),
[figures](results/qa_v1/figures/accuracy.png), and
[verification summaries and provenance](results/qa_v1/MANIFEST.md).
Datasets, fitted models, predictions, and full audit evidence remain local.
