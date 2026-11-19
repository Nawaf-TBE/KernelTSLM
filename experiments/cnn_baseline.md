# Frozen primary L1 CNN baseline

The unweighted seed-7 residual CNN at epoch 7 is selected by validation macro-F1
over the 11 training-supported classes (0.9459247843). The weighted comparison
scored 0.9434112005. This is validation performance, not test performance.

See `docs/phase3_cnn_baseline.md` for the full controlled protocol, per-class and
temporal diagnostics, nondeterminism limitations and resource observations.
The read-only bundle is `outputs/phase3/cnn/frozen/`; verify `bundle.json` and
`manifest.json` before loading `baseline.pt`. No test/QA/L2/OpenTSLM work is
implied or authorized by this frozen artifact.
