# QA baseline results — draft for a future README update

Frozen QA v1 validation: primary logistic regression achieved **93.375% comparison**
and **62.289% temporal** accuracy; the counter-pair shortcut achieved **56.125%**
and **61.769%**. The primary temporal improvement is not established by the paired
whole-trace interval. Calculation-aware features reached **98.125% / 89.597%**,
but explicitly include answer-sufficient maxima/peak locations. Deterministic
calculation answered every task correctly; this is an arithmetic control, not AI
reasoning. A training-median numerical predictor had MAE **0.553243** versus zero
for direct calculation. Independent feature/model/metric replay verifies results
with qualifications. No OpenTSLM fine-tuning or official test evaluation occurred.

Aggregate summaries are available on the QA review branch; full artifacts remain local. See phase4_qa_baselines.md for
protocol, confidence intervals, reversal controls, and limitations. Do not publish
generated datasets or claim learned temporal reasoning from these controls.
