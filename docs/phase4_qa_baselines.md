# Phase 4 Task 2 — QA baselines and independent verification

## Objective and frozen benchmark

Establish interpretable reference methods before OpenTSLM fine-tuning, separating
counter-identity shortcuts, learned summary features, and explicit calculation.
These are **official-validation** results on frozen QA v1: 16,652 training and
2,369 validation questions (800 statistics, 800 comparison, 769 temporal).
During experiment construction there was no QA regeneration, threshold change,
operation-label feature use, CNN retraining, official test evaluation, or publication.

The [QA v1 specification](phase4_qa_v1.md) defines normalized units and half-open
intervals. Train JSONL SHA256 is
`d63a0544e7ae3e3029d1af42c4a9a9f0d10308a8f4fcab7f29eecd816f298d41`;
validation SHA256 is
`e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df`.
The complete QA artifact manifest is checked before fitting and auditing.

## Protocol and implementations

The [versioned configuration](../configs/phase4_qa_baselines_v1.json) was written
before fitting/evaluation and copied, hashed, into the output protocol.
Only training questions fit rules, scaler moments, and model coefficients.
There is no hyperparameter search, early stopping, validation-selected variant,
or question-level training/holdout split. Seed 7 and a single native math thread
are fixed. The existing isolated train/validation TimeF loader and unchanged
training-only normalization are reused. Its reader/row-group/NPZ guards prohibit
mixed-release and test-array access. Training/validation trace sets are disjoint.
The adapter validates operation-label consistency internally, but QA feature
extraction never receives operation labels or label-derived masks.

[Core implementation](../src/kernel_xray/qa_baselines/core.py):

- **A — Training-majority:** separate categorical mode per task. Ties retain the
  first encountered training answer, matching the frozen QA quality diagnostic.
  Statistics use the training median answer **-0.19676105341530675**, a constant
  estimator, not an arithmetic solver.
- **B — Counter-pair lookup:** ordered (A,B) counter-ID pair to training answer
  mode, separately per categorical task. Unseen pairs fall back to A. No time-
  series measurements are used. A/B answers refer to question presentation order.
- **C — Primary logistic regression:** per queried counter, interval mean,
  population standard deviation, first value, and last value; A-minus-B
  differences of those four quantities; interval length/512; and two ordered
  24-dimensional one-hot channel IDs. **61 features**, 13 continuous plus 48 ID
  indicators. No interval maximum, minimum, or peak location is supplied.
- **Calculation-aware LR control:** add min, max, and first peak location as a
  fraction of length minus one for each counter, plus their A-minus-B
  differences. **70 features**, 22 continuous plus 48 IDs. Maximum differences
  and relative unique-peak locations are answer-sufficient for these tasks.
  This control is not evidence of independently learned arithmetic or reasoning.
- **D — Direct calculation:** compute the interval mean, both maxima, or unique
  peak ordering directly from normalized measurements and safe structured
  parameters. Frozen ambiguity thresholds apply; no saved operands or answers
  enter calculation. This is a deterministic control, not a trained AI model.

For each categorical task and feature variant, continuous features receive a
training-fitted `StandardScaler`; one-hot IDs pass through unchanged. Logistic
regression uses `C=1`, `solver=lbfgs`, `max_iter=3000`, `tol=1e-8`, random state 7,
L2 regularization and default intercept, without class weighting. Installed
scikit-learn is 1.9.1. All four fits converged (67/111 comparison, 414/336 temporal
iterations for primary/aware). Repeating each fit produced bitwise-identical
coefficients, intercepts and scaler parameters.

All feature functions take only `[24,512]` float32 counters and the strict
allowlist `type/channels/start/end`. IDs, trace identity, split, question answers,
oracle operands, and label-bearing metadata are excluded. The question string is
verified by the frozen QA validator but not parsed by these baselines. This task
therefore evaluates structured-parameter QA, not natural-language understanding.

## Verified categorical validation results

Accuracy intervals are pooled whole-trace percentile 95% intervals.

| Method | Comparison accuracy [95% CI] | Balanced accuracy | Macro-F1 | Temporal accuracy [95% CI] | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority | .511250 [.496250, .527500] | .500000 | .338296 | .375813 [.352789, .401356] | .333333 | .182105 |
| Counter pair | .561250 [.528750, .593750] | .560437 | .560205 | .617685 [.577120, .655135] | .617828 | .616977 |
| Primary LR | .933750 [.917500, .948750] | .933294 | .933637 | .622887 [.583227, .660574] | .622476 | .621536 |
| Aware LR control | .981250 [.971250, .990000] | .980987 | .981229 | .895969 [.870248, .919737] | .890975 | .898700 |
| Direct calculation | 1.000000 [1, 1] | 1.000000 | 1.000000 | 1.000000 [1, 1] | 1.000000 | 1.000000 |

The counter-pair results **exactly reproduce** the original 56.125% comparison
and 61.7685% temporal QA v1 diagnostics. No answers were changed to obtain a match.
`metrics.json` includes integer confusion matrices and per-answer precision,
recall, F1, and support for every condition; `metrics_independent.json` matches.
Comparison evaluates actual supported classes A/B; Tie is supported by the
oracle but has no v1 examples. Temporal classes are No/Same bin/Yes. There is no
combined numerical-and-categorical score.

### Paired improvements over the counter-pair shortcut

| Contrast | Comparison accuracy difference [95% CI] | Temporal accuracy difference [95% CI] |
|---|---:|---:|
| Primary minus lookup | +.372500 [.337469, .408750] | +.005202 [-.021249, .030832] |
| Aware minus lookup | +.420000 [.387500, .452500] | +.278283 [.238153, .320975] |

The primary temporal model's small point improvement is **not established** by
this interval. Primary comparison features help substantially within this cohort,
but mean/variance/endpoints can act as maximum proxies, especially for flat or
correlated signals. High comparison accuracy is not proof of general arithmetic.

## Numerical validation results: statistics

| Method | MAE | Median absolute error | RMSE | Within frozen tolerance |
|---|---:|---:|---:|---:|
| Training-median constant | .553243 | .223490 | .966459 | 0% |
| Direct calculation | 0 | 0 | 0 | 100% |

Constant-estimator 95% intervals: MAE [.486422, .625120], median absolute error
[.209019, .277566], RMSE [.828415, 1.106033], and within-tolerance fraction [0,0].
Direct errors are zero and tolerance coverage one in every whole-trace replicate.
Paired direct-minus-constant error intervals are the negated constant intervals;
coverage improvement is +1. The numerical tolerance remains
`abs(pred-ref) <= 1e-6 + 1e-5*abs(ref)` in normalized units. Direct calculation's
100% is expected mathematical consistency, not AI reasoning performance.

## Uncertainty

5,000 bootstrap replicates, seed 7, resample all **160 whole validation traces**,
carrying each trace's questions together. The same trace multiplicities are used
for all tasks, models and paired differences. Metrics are recomputed from pooled
counts/errors, never by treating 2,369 individual questions as independent.
Categorical classes remain fixed; resampled absent classes get zero recall/F1.
No supported class is absent in these observed categorical resamples. The
uncertainty report includes accuracy, balanced accuracy, macro-F1, per-answer
precision/recall/F1 intervals, numerical metric intervals, and paired differences.

These are conditional intervals for fixed fits and this selected QA cohort. They
do not include training-seed variability or demonstrate unseen-workload/hardware
transfer. Shared workload families can violate even trace-level independence.

## Shortcut and temporal-order controls

For a channel-ID-only sensitivity control, replace every continuous feature with
its **training scaler mean** (zero after standardization), retaining channel-ID
indicators. No refitting occurs; this is an off-distribution feature ablation.

| Method | Comparison original / identity-only | Temporal original / identity-only |
|---|---:|---:|
| Primary LR | .933750 / .488750 | .622887 / .617685 |
| Aware LR | .981250 / .488750 | .895969 / .375813 |

This shows dependence on measurements, particularly for comparison and aware
models. Matching an identity-only accuracy to lookup does not imply identical
predictions. It also does not eliminate identity/measurement interactions.

### Reverse-time control

Reverse both queried channels **within the selected interval**, preserving the
rest of the trace, channel identity, question text and interval bounds. Recompute
features and model predictions from altered values, without refitting or reading
original oracle intermediates. Independently calculate the altered signal's
correct answers. Every altered example remains mathematically well-defined.

Comparison maxima and answers are invariant to reversal. Unique temporal peak
locations map to `start+end-1-original_peak`; Yes/No swap while Same bin stays.
Thus 62.419% of temporal reference answers change. Report both target conventions:

| Temporal method | Original input / original labels | Reversed input / original labels | Reversed input / independently altered labels |
|---|---:|---:|---:|
| Majority | .375813 | .375813 | .375813 |
| Counter pair | .617685 | .617685 | .340702 |
| Primary LR | .622887 | .607282 | .364109 |
| Aware LR | .895969 | .365410 | .849155 |
| Direct calculation | 1.000000 | .375813 | 1.000000 |

Primary temporal predictions change on 7.542% of questions; its reverse-minus-
original accuracy interval against original labels is [-.032553, .001282].
There is **no clear evidence here of useful primary temporal-order contribution**
beyond shortcuts. The aware control changes on 54.226% and largely tracks altered
answers (84.915%); its features explicitly supply peak order. Direct calculation
tracks all changed answers exactly. Lower original-label accuracy by itself is
not evidence of meaningful reasoning: labels have deliberately become stale.

For comparison, direct calculation remains exactly invariant. Primary/aware
classifiers change 1.75%/.625% of predictions because endpoint and peak-position
features are order-sensitive nuisance inputs, even though the target is invariant.
Their reverse-input accuracies are .936250/.982500, with paired changes spanning
zero. Neither this small increase nor temporal sensitivity is a validation tuning
criterion. No QA v1 redesign or new hyperparameter choice followed these results.

## Independent verification

**Verdict: VERIFIED WITH QUALIFICATIONS.**

The [independent audit implementation](../src/kernel_xray/qa_baselines/audit.py)
uses scalar `math.fsum`, scalar population standard deviation, Python min/max and
explicit peak searches to reconstruct features from authorized tensors. It does
not reuse the engine's extractor or metric functions. It reopens all training
and validation views in a fresh process, validates every frozen QA identity,
channel mapping and reference, and compares feature matrices within `1e-10`.
Training scaler moments/sample counts and training majority/pair rules are
recomputed independently. Frozen models are replayed by manually applying scaler
moments and coefficient/intercept matrices; all predicted classes match exactly.
Manual Python confusion counts and scalar errors reproduce every saved metric,
including altered-label and identity controls, at `1e-12` tolerance.

Protocol/source/model/rule hashes are checked; QA JSONL and all frozen QA artifact
hashes remain unchanged. Tests mutate target/oracle/identity fields, assert
feature invariance, reject unsafe parameter keys, check interval boundaries,
tie/peak semantics, train-only scaler state, deterministic fitting and whole-
trace paired bootstrap behavior. Preservation hashes cover all captured earlier
data, views, CNN and QA artifacts, source/configuration/tests/docs/results.

Focused tests: **16 passed**. Full relevant regression: **110 passed, one CUDA-availability skip**, zero failures/errors.

The first attempt's preservation guard incorrectly included its own growing log.
It stopped after fitting/evaluation; no historical file change was detected.
The complete attempt and executed driver are retained under
`initial_attempt_preservation_log_error/`. The corrected run uses the same fixed
hyperparameters and excludes its own logs. This was an inventory correction,
not validation-guided model tuning; both attempts' fitted results can be compared.

## Files and commands

All new outputs are under `outputs/phase4/qa_baselines_v1/`:

- `protocol.json`, `rules.json`, `models.pkl`, `model_hashes.json`, `repeatability.json`.
- `predictions.json`, `metrics.json`, `metrics_independent.json`.
- `uncertainty.json`, `statistics_paired_uncertainty.json`, `shortcuts.json`, `ablation.json`.
- `verification.json`, `COMPLETE.json`, preservation inventories, tests and logs.
- Feature caches are counter-derived arrays only; question IDs are bookkeeping.
  Targets remain separate. Treat `models.pkl` as trusted-local pickle, not an
  untrusted import or a published artifact.

From the prepared workspace, use the existing TimeNet environment (no new install):

```sh
export PYTHONPATH="$PWD/src:$PWD"
export UV_CACHE_DIR="$PWD/.cache/uv"
UV="$PWD/.worktrees/timenet-kernelsight/.venv/bin/uv"
# First-time fitting only: refuses existing frozen models/inventory.
"$UV" run --project .worktrees/timenet-kernelsight python scripts/run_qa_baselines.py
# Safe repeat audit: no fitting or QA regeneration.
"$UV" run --project .worktrees/timenet-kernelsight python scripts/run_qa_baselines.py --audit
"$UV" run --project .worktrees/timenet-kernelsight python -m pytest tests -q -rs --junitxml=outputs/phase4/qa_baselines_v1/regression.xml
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python scripts/plot_qa_baselines.py
"$UV" run --project .worktrees/timenet-kernelsight python scripts/finalize_qa_baselines.py
```

Full tests involving MPS/shared-memory may require host execution. Do not rerun
QA generation or original verifiers that overwrite historical results.

## Validation figures

Plots use the same saved, independently verified metrics and uncertainty report.
Generated files are local and not automatically published:

- [Accuracy and trace intervals](../results/qa_v1/figures/accuracy.png)
- [Learned-model confusion matrices](../results/qa_v1/figures/confusions.png)
- [Statistics error distribution](../results/qa_v1/figures/statistics_errors.png)
- [Reverse-time ablation](../results/qa_v1/figures/temporal_ablation.png)
- [Paired differences](../results/qa_v1/figures/paired_intervals.png)

## What the experiments establish

Simple counter summaries help maximum comparison on this QA cohort. The primary
model does not establish useful temporal reasoning beyond the counter-pair
shortcut. Explicit peak-location features and direct calculation establish
measurement consistency and useful controls, not independently learned reasoning.
No natural-language parsing, broad mathematical reasoning, operation recognition,
overlap detection, causal explanation, OpenTSLM ability, official test performance,
or cross-hardware generalization is established. Results inherit QA v1's narrow
templates, normalized-unit interpretation, category balancing, unique-peak
selection bias, repeated workloads, and single-seed limitations.

## GitHub review exports

[Aggregate metrics, copied verdicts, and provenance](../results/qa_v1/MANIFEST.md)
are published on the dedicated review branch. Full experiment artifacts remain
local; the original pre-publication document is preserved locally.
