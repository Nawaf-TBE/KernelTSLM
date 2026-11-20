# QA v1 reproducibility and research artifacts

## Quick inspection

Start with [QA definitions](phase4_qa_v1.md), [baseline results](phase4_qa_baselines.md),
[aggregate figures](../results/qa_v1/figures/accuracy.png), and the
[artifact manifest](../results/qa_v1/reproducibility_manifest.json).
The branch includes source/tests/configuration, research documentation, six figures,
aggregate results, four portable fitted LR models, and per-condition prediction
records. No raw GPU arrays, QA JSONL, fitted pickle, or compiled views are bundled.

## Artifact-only verification — no dataset or training

From the repository root, with Python 3.9 or newer:

```sh
python3 scripts/verify_published_qa.py
```

This command uses the standard library plus the Git CLI for tracked-path checks. It validates model JSON schema, hashes,
feature order, numerical finiteness, prediction schemas and unique IDs, the derived
validation-question index, target consistency, and all original/reversal/identity
condition metrics. Confusion counts, accuracy, balanced accuracy, macro-F1,
per-class precision/recall/F1, MAE, median absolute error, RMSE and tolerance coverage
are recalculated using independent scalar arithmetic. Failures raise and exit
nonzero. It does not import pickle or run fitted Python estimators.

IDs are checked against a hash-provenanced index exported from the frozen QA.
Without the excluded original QA files, this cannot independently prove that index
corresponds to the original dataset; the local export checked it against every
frozen question. Full signal-derived answer verification requires authorized data.
No artifact-only claim substitutes for inference replay or original audit evidence.

## Data-sharing review

The hash-matched KernelSight card declares Apache-2.0 and attributes KernelBench
(MIT) and CUTLASS (BSD-3-Clause). The pinned distribution contains no LICENSE/NOTICE
bundle or exact collection-time dependency commits. Full notice completeness is
unresolved; private GitHub hosting is still redistribution. Therefore **neither
train nor validation QA JSONL is published**. Their contents were inspected:
19,021 records, relative `kernels/...` trace IDs, no embedded raw tensors,
credentials or personal filesystem paths, and file sizes of 14,083,869 bytes (train) / 1,995,743 bytes (validation). Exclusion is a
licensing-completeness decision, not a size or sanitization problem. Frozen bytes
are untouched. See [source/notice review](source_provenance_and_license.md).

Primary references: [pinned dataset card](https://huggingface.co/datasets/williamhtan/kernelsight/blob/d8cd8e804e5b619aadbb082a1fb3a8791b60a680/README.md)
and [Apache-2.0 redistribution conditions](https://www.apache.org/licenses/LICENSE-2.0).
This publication does not claim to resolve missing collection-time notices.

Published prediction records contain only hashed question IDs and derived scalar
or categorical evaluation facts, not original question strings, trace paths,
counter arrays, collection metadata or oracle operands. They are research evaluation
exports, not a repackaged KernelSight trace dataset. Numerical model exports are
our fitted coefficient/scaler parameters, not copies of third-party source or raw
measurements. No blanket license for underlying data is inferred from these exports.

## Portable models

[models/qa_v1](../models/qa_v1/) contains comparison/temporal × primary/aware JSON.
Each records coefficients, intercepts, class order, exact feature names/order,
training scaler means/scales/variance/sample count, fixed one-hot preprocessing,
estimator configuration, source model/protocol/configuration/code SHA256, and
normalization SHA256. No executable serialization is used. The original trusted-
local pickle remains external; only the explicit export script ever reads it.

```python
from kernel_xray.qa_baselines.portable import PortableModel
model = PortableModel.load('models/qa_v1/temporal_primary.json')
# feature_matrix must use the documented 61-column original order:
# classes = model.predict(feature_matrix)
# probabilities = model.predict_proba(feature_matrix)
```

Prediction methods require NumPy; artifact schema/metric verification does not.
The independent local replay used all 160 validation traces, exact cached original
features and independently reconstructed scalar features. All classes matched;
probabilities agreed at predeclared rtol=atol=1e-12. The copied
[model-equivalence evidence](../results/qa_v1/model_equivalence.json) records maxima
and original model hashes. Degenerate/tied decisions use original class order.

## Full reproduction — external prerequisites

A fresh clone alone is insufficient. An authorized collaborator needs:

1. KernelSight v4 from its official repository, pinned to
   `d8cd8e804e5b619aadbb082a1fb3a8791b60a680`. Obtain it legitimately from the pinned
   dataset link above and retain source notices; do not use an invented mirror.
   Arrange the unchanged `kernels/`, `splits/`, `tools/` at
   `data/kernelsight_v4/` and preserve root compatibility aliases as documented in
   [layout/model records](model_records.md). A source snapshot normally also has
   its card/manifest; do not overwrite this project's README with the source card.
2. The separately maintained TimeNet development worktree at
   `.worktrees/timenet-kernelsight`, based on
   `9bd1613019ee61fbc52e02106a1a45990ead7a54`, **plus the actual unsubmitted KernelSight
   connector and two writer fixes**. Those changes are not in upstream main or
   this repository. Request the owner's matching approved snapshot/patch and uv
   lock/environment; no public patch/download or namespace approval is invented.
   The upstream repository is https://github.com/OpenTSLM/TimeNet but that base
   commit alone cannot provide this connector. See [connector record](timenet_connector_task2.md).
3. Verified training-only normalization at
   `outputs/phase1/task7/train_statistics.json`, SHA256
   `66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`.
   Obtain the original from the owner or use the existing training-only Phase 1
   verifier, then require this hash. Do not fit on validation or replace the frozen
   statistics with per-trace scaling.
4. Separate train/validation TimeF views. The following existing commands are for
   a prepared authorized connector environment and fresh absent output directories:

```sh
export PYTHONPATH="$PWD/src:$PWD"
export UV_CACHE_DIR="$PWD/.cache/uv"
UV="$PWD/.worktrees/timenet-kernelsight/.venv/bin/uv"
# Builders refuse existing paths; no test partition is requested.
"$UV" run --project .worktrees/timenet-kernelsight python scripts/prepare_phase3_timef_views.py --split train
"$UV" run --project .worktrees/timenet-kernelsight python scripts/prepare_phase3_timef_views.py --split val
```

These builder commands were inspected and shell-syntax checked, **not executed
again for publication**. They require approved memory/disk resources and source
permission. The existing [pipeline guide](phase3_baseline_data_pipeline.md) details
mechanical verification and the earlier mixed-row-group issue. Never use the mixed
full artifact as a substitute for the strict views.

The recorded environment is Python 3.13.4, NumPy 2.5.0, PyTorch 2.13.0,
TimeNet 0.12.0, PyArrow 24.0.0, scikit-learn 1.9.1. Use the matching external uv
lock; root Phase 1 requirements target a different Python 3.9 environment and
must not be installed over this environment. There is no fabricated fresh-clone
installation recipe while the unpublished connector dependency remains external.

Obtain the exact frozen QA JSONL privately subject to permissions, or follow the
existing [deterministic generation/verification workflow](phase4_qa_v1.md) locally.
Generation is optional external reproduction, not part of artifact inspection;
no generation was performed for this publication. Require the frozen hashes in
[MANIFEST.md](../results/qa_v1/MANIFEST.md), rather than rewriting records to match.

Once authorized data, normalization, views, and frozen QA are present, this
read-only command replays the portable models, without pickle or model fitting:

```sh
export PYTHONPATH="$PWD/src:$PWD"
export UV_CACHE_DIR="$PWD/.cache/uv"
.worktrees/timenet-kernelsight/.venv/bin/uv run --project .worktrees/timenet-kernelsight python scripts/replay_portable_qa.py
```

The command was actually run successfully for this publication. It compares all
3,138 categorical model predictions (two models × 1,569 questions) against the
published originals. Original QA validation remains available separately in the
prepared environment; historical preservation inventories may flag intentional
README/link changes—do not replace those inventories to suppress differences.

## Interpretation

Structured parameters are supplied; language parsing is untested. All counters
are train-normalized, not physical-unit comparisons. Identity shortcuts are
material; primary temporal LR has not demonstrated useful reasoning beyond them.
Aware LR has answer-sufficient features, and direct arithmetic is not learned
reasoning. Workload overlap, single seed and selected unique peaks limit inference.
Official test data remains unevaluated by QA. No OpenTSLM experiment is claimed.
