# Evaluate a saved model on a new cohort

Use `evaluate` to measure a fixed model's research scores without refitting weights,
imputation, scales, thresholds or aggregation rules:

```bash
xquantum evaluate --model /secure/runs/frozen-model \
  --manifest /secure/external-cohort/manifest.jsonl \
  --development-manifest /secure/original-cohort/manifest.jsonl \
  --output /secure/evaluations/external-001.json --seed 42
```

The command reads local data only. The original model weights and training report
are unchanged. The output path must be new. A report is written only after the
cohort, overlap, quality and metric checks all succeed.

## Reference data and overlap checks

The development manifest must match the SHA-256 recorded in the model's training
report, including its exact bytes and line endings. Keep the original manifest and
images in their original relative layout. The command audits their integrity and
compares the new cohort against **every original record**, including validation and
test records, so previously used development/test cases cannot be relabeled as an
independent cohort.

It rejects shared patient IDs, shared source groups, matching image-file hashes,
matching decoded pixel hashes, and perceptual-hash distances of four bits or less.
These are conservative screens: a suspected near duplicate stops the run and needs
source reconciliation. Do not change identifiers to defeat the check. This cannot
prove that different identifiers or unrecognized reuploads belong to different
patients, and patient identities may be incomplete. In particular, the public
Wikimedia model cannot establish patient disjointness because its source patient
identities are unknown. That limitation is explicitly recorded in the report.

These checks do not establish an independent clinic, device or time period. Plan
those aspects and the sample size in the study protocol. The hash comparisons are
intended for the small research cohorts supported by this baseline; cross-cohort
perceptual comparison is quadratic in image counts.

## Cohort requirements

All new records must have `split: test`, a specialist-reviewed label, reviewer ID,
known patient group, source group, known photo modality, usable source rights, and
matching image checksums. Labels must be within the saved model's class scope.
Unknown diseases and controls absent from that model cause a failure rather than
being silently excluded. Missing modeled classes produce undefined metrics and
warnings under the existing evaluation rules.

Every image must pass the same basic quality gates used for analysis. A single
quality failure stops the entire evaluation without metrics; it is not omitted to
improve the apparent performance. Review the cohort according to prespecified
acquisition/exclusion rules rather than choosing exclusions from score outcomes.

Optional clinical context is processed only if the frozen model was trained for
that input. Missing values remain unknown and use its original preprocessing.
Existing licence and restricted-source checks apply to the external cohort too.
The report flags restricted-source use for review before publication.

## Results and interpretation

The JSON report binds the saved weight ID, development manifest hash and external
manifest hash. It records aggregate image-level/per-condition metrics, primary
patient-level metrics, and patient-bootstrap intervals where supported. See
[EVALUATION.md](EVALUATION.md) for aggregation, undefined values and interval assumptions.
It contains no patient IDs, filenames, individual predictions or clinical values.

These are metrics of forced class rankings from an uncalibrated research model.
They are not diagnostic performance estimates for a deployed medical system: all
actual diagnostic responses still abstain, and `clinically_validated` stays false.
Freezing weights alone does not establish clinical validity. Avoid repeated model
selection against this cohort and preserve it as a genuinely held-out analysis.

The software tests use generated fixtures to establish behavior. No new clinical
cohort or clinical validation result accompanies this feature. The original
public training model and ten-photo ZIP are unchanged.
