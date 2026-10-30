# Patient-level evaluation

Reviewed research training now writes two analyses under each validation/test split
in `report.json`. The top-level fields retain image-level results for compatibility;
`primary_analysis` explicitly selects `patient_level` for the primary analysis.
No metrics are emitted for the public bootstrap cohort, whose patient IDs are unknown.

## Unit of analysis

Average each patient's class-score vectors across all supplied views, then take the
highest average score as that patient's prediction. Every patient receives equal
weight regardless of photo count. This is a specified research aggregation rule,
not a clinically validated fusion strategy. A patient must have one adjudicated
primary label within a split; conflicting labels fail evaluation. Multi-condition
or changing diagnoses require a different task definition before inclusion.

Image-level and patient-level results include a confusion matrix with explicit
`class_order`, class support, sensitivity (recall), specificity, precision, F1,
balanced accuracy, macro F1 and log loss. Scores remain uncalibrated. A missing
positive denominator makes sensitivity null; a missing negative denominator makes
specificity null; no predicted positives makes precision null. F1 uses zero for
an empty class, and macro F1 averages across the full fixed class list. Balanced
accuracy is null if any modeled class is absent, with an explicit warning.

## Confidence intervals

The patient-level analysis includes deterministic 95% percentile intervals for
balanced accuracy, macro F1 and log loss. It resamples independent patient predictions
within true-label strata, with replacement, preserving each observed class count.
It uses 1,000 replicates, the training seed, and NumPy's linear percentile method.
All views of a patient have already been combined and never act as independent
bootstrap samples. Reports contain aggregate counts and metrics, not patient IDs
or individual predictions.

Intervals are withheld if any class has fewer than two independent patients.
That is a computational guard, not an adequate clinical sample-size target. Even
when emitted, intervals can be unstable or degenerate for small or perfectly
separated cohorts. They condition on the observed case mix and a fixed fitted model;
they exclude training uncertainty, diagnosis uncertainty, selection bias and domain
shift. They do not provide evidence of clinical validity. Per-condition intervals,
calibration, subgroup analysis and external validation remain future work.

## Running

Use the existing reviewed-data workflow; no additional switches or dependencies:

```bash
xquantum train --manifest /secure/dataset/manifest.jsonl \
  --output /secure/runs/patient-evaluation --epochs 50 --seed 42
```

Inspect `held_out_metrics.validation.patient_level` and
`held_out_metrics.test.patient_level` in the resulting report. Plan the cohort,
patient grouping and views before fitting. Do not tune aggregation or models on
the final test results. The included bootstrap weights have not been retrained or
assigned fabricated held-out performance as part of this software improvement.
