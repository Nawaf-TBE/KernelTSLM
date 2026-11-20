# Verified QA v1 publication manifest

Review branch: `feature/qa-v1-verified-baselines` in private `Nawaf-TBE/kernel-xray`.
Base main: `e50bdcc1d5d56a4960cc3e7dca241ad68e0c8d8a`. The publication commit
is available from this file's Git history; its own SHA is omitted to avoid a
self-referential commit. No merge or PR is implied.

## Frozen references

- Train: 16,652 questions; SHA256 `d63a0544e7ae3e3029d1af42c4a9a9f0d10308a8f4fcab7f29eecd816f298d41`.
- Validation: 2,369; SHA256 `e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df`.
- Total: 19,021; independent answer agreement 100%; exact regeneration passed.
- Normalization: original channel-wise training population statistics, SHA256
  `66e32bb7f1390c0dcd86f04ab6aea5a25a975af5577f0bdc3f94e35e4a42a1c1`.
- Original baseline verdict: **VERIFIED WITH QUALIFICATIONS**; all independent
  feature/model/metric replays passed. Full baseline regression: 110 passed,
  one CUDA skip. Original full evidence remains local.

## Published scope and provenance

The original exports contain aggregate summaries and six aggregate plots. Full QA JSONL,
TimeF views, original full per-question predictions, feature caches, fitted model pickle, raw
data, environments, and real-trace measurement illustrations remain local.
No model or dataset redistribution license is asserted by this publication.
Source artifact paths and byte hashes are in [provenance.json](provenance.json).
Copies preserve bytes; derived JSON exports retain source and output hashes.
The original method document was snapshotted locally before publication link edits.

This summary is **not a replacement for the full independent audit evidence**.
The copied [QA verification](qa_verification.json) and
[baseline verification](baseline_verification.json) refer to that local evidence;
paths into outputs/ are local artifact references, not bundled files.

## Scientific limitations

Structured question parameters are supplied; natural-language parsing is not
measured. Counters use training-derived normalization, not physical units.
Counter-identity shortcuts are material. Primary temporal LR does not establish
useful temporal reasoning: +0.52 percentage points over lookup, paired 95% CI
[-2.12,+3.08]. Calculation-aware LR receives answer-sufficient maxima/peak
features. Direct calculation is an oracle-style control, not learned reasoning.
Shared workload families, single-seed fitting, narrow templates, and peak-selection
bias limit generalization. Official test data remains unevaluated by QA work.
No OpenTSLM training or evaluation is claimed.

## Reproduction

See [QA definitions](../../docs/phase4_qa_v1.md) and
[baseline protocol](../../docs/phase4_qa_baselines.md). Experiments require
owner-authorized local data, isolated views, and the separate TimeNet environment;
a clone alone cannot replay them. No download URLs are invented. Do not regenerate
frozen QA to inspect the results; use the safe audit command on prepared data.
GitHub-readable source, metrics, confidence intervals, and figures permit review
without private data, but full raw replay requires access to local evidence.
Historical source-document hashes refer to pre-publication documents; publication
link changes do not change experimental code or results. Historical preservation
inventories also include the old README: publication intentionally changes that
file. Its original snapshot is retained locally under
`outputs/phase4/qa_publication/original_documents/`. Do not replace old inventories
to suppress that expected documentation difference.

Run `python3 scripts/verify_qa_publication.py` in a clone to check published
artifact hashes and links without data access. Add `--local-artifacts` only in
the original prepared workspace for frozen dataset/CNN/output byte checks.

## Published file SHA256

| File | SHA256 |
|---|---|
| `baseline_verification.json` | `a51ae51be3e3e085406a1276e3688395a2c340cb7899b5ca03607a866ae6fd9a` |
| `categorical_metrics.json` | `79331c3a82fb178c67d7ee239d8d60a29a97083d4675eda964ab3e0679586b9b` |
| `confidence_intervals.json` | `8cd78d0aa005943defb73843793e6652b322ed68779d9e3f30c487ad0e39ff5c` |
| `dataset_summary.json` | `c69ecbc923d21ea7f3994b24f90bc29dd2f91dc29ef2bdea0480a389afd45679` |
| `figures/accuracy.png` | `3edd7c7de4e8e6cee0e1f6de656dea13f8ba7e3eac8b81f6cbddd6f8d0180917` |
| `figures/confusions.png` | `c6a20dde3700dfc44b55735164df3b3c3bc33b201cbc85be956562dd179c4fa3` |
| `figures/paired_intervals.png` | `a156d57ec5eb658244959f31fc2b8272d21d9664458644e99e78ff6d23918cb7` |
| `figures/qa_distributions.png` | `3aeb8a0f9b9bf888f6d38bffafcd5e143af5a7f6f52e14b626af7ae642b1a50a` |
| `figures/statistics_errors.png` | `192a69bbc73132f6f04a2d102f1317386f9cf8ee80b9645501eeaee82c18cb74` |
| `figures/temporal_ablation.png` | `804377b234cd69e7ec82bf485975ac061155a4897e0eacc47784922f1beed08d` |
| `provenance.json` | `8567f06bdbf1cebb16b901d05c873d9bc4f4b1b72c6a6ea9cbff3d0447ac673c` |
| `qa_verification.json` | `8dd0aa48a4ecf0f3df5913b8fcd61bcf5b93447049c9ea6d0d106eb85f40a499` |
| `shortcut_diagnostics.json` | `e3e8cd7ed3f4f34b8154a356d24aa635342733fb5fda142852298a566d7196e8` |
| `statistics_metrics.json` | `916ffbacfd105cfcf64122cd0f4359a11b2dff644f461ee29c627089c8db56a6` |
| `temporal_ablation.json` | `1d21d270128c4592fd465821e5abe24773f8418199d62c535b7c2d4b0289b925` |

## Portable reproducibility extension

[Reproducibility manifest](reproducibility_manifest.json) records the SHA256 of
four [safe JSON LR models](../../models/qa_v1/),
[minimal prediction records](predictions.json), [all-condition metrics](condition_metrics.json),
[derived question-ID index](question_index.json), and [model equivalence](model_equivalence.json).
Original trusted-local model SHA256:
`1da48cc45327ccc4071a64a99bbf526260972ad78d6d5d3b657600a2d52a670e`.
The prediction source is retained locally with its source hash in that manifest;
the export strips trace identifiers and preserves condition, ID, reference and
prediction only. No raw arrays, oracle operands or question text are included.

QA JSONL remains excluded: collection-time third-party notice completeness is
unresolved despite the source card's Apache-2.0 declaration. Private hosting does
not remove redistribution obligations. Original records were inspected for size,
schema and sensitive paths but never rewritten. Scalar prediction facts and fitted
parameters are research evaluation exports, not a repackaged trace dataset.

Run `python3 scripts/verify_published_qa.py` without data or third-party packages.
It recalculates every condition's categorical and numerical metrics. Dataset-ID
alignment is checked against a derived frozen index; external original QA is
required to independently authenticate that index against source bytes.
[Full reproduction guidance](../../docs/REPRODUCIBILITY.md) separates this check
from signal-based inference. All prior scientific qualifications still apply.
