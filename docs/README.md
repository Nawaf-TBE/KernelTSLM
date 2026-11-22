# Documentation

Method notes for the CNN baseline and for bringing KernelSight into TimeNet. The multi-task language-model work is
described in the [main README](../README.md) and in `usecases/<task>/SPEC.md`.

Files under `outputs/` that these notes cite are local artifacts and are not in the repository; the copied figures,
aggregate results and checksums are in `results/cnn/`.

## CNN baseline

| File | Contents |
|---|---|
| [cnn-baseline.md](cnn-baseline.md) | Results summary: validation macro-F1, accuracy, qualifications |
| [phase3_cnn_baseline.md](phase3_cnn_baseline.md) | Training protocol, model selection, per-class and temporal diagnostics |
| [phase3_cnn_independent_audit.md](phase3_cnn_independent_audit.md) | Independent replay of the frozen checkpoint and its verdict |
| [phase3_baseline_data_pipeline.md](phase3_baseline_data_pipeline.md) | Train and validation TimeF views the CNN reads |
| [model_records.md](model_records.md) | `KernelSightDataset`, the PyTorch loader in `src/kernel_xray/dataset.py` |
| [visualization.md](visualization.md) | Trace plots from `scripts/visualize_traces.py` |

## KernelSight in TimeNet

The connector itself is in TimeNet ([OpenTSLM/TimeNet#835](https://github.com/OpenTSLM/TimeNet/pull/835)).

| File | Contents |
|---|---|
| [timenet_integration_plan.md](timenet_integration_plan.md) | Initial design for the connector |
| [timenet_decision_review.md](timenet_decision_review.md) | Review of that design; overrides the plan where they differ |
| [timenet_implementation_spec.md](timenet_implementation_spec.md) | API choices for the implementation |
| [timenet_connector_task2.md](timenet_connector_task2.md) | Connector implementation, commands and limitations |
| [timenet_memory_optimization.md](timenet_memory_optimization.md) | Memory profiling of the conversion |
| [timenet_full_conversion_validation.md](timenet_full_conversion_validation.md) | Full 1,444-trace conversion and integrity checks |
| [source_provenance_and_license.md](source_provenance_and_license.md) | Check that the local data matches the published KernelSight release |
