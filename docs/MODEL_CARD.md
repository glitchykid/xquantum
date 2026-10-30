# Model card: initial research baseline

**Intended users:** researchers working with dermatologists and trichologists.
**Intended task:** verify a reproducible image ingestion, training and inference path.
**Not authorized by evidence:** patient diagnosis, screening, treatment selection,
severity grading or autonomous triage. Every output returns `status: abstain`.

## Architecture

Pillow decodes and orients images. A 128×128 padded image is divided into a 4×4
grid. Each cell contributes 18 RGB histogram features and 9 gradient-orientation
features: 432 visual features total. A class-balanced, L2-regularized SGD logistic
classifier learns the output weights. There is no scalp segmentation, lesion
localization, pretrained network, causal inference or hair-density measurement.

Optional reviewed-data experiments concatenate structured clinical values and missingness
indicators. Training-only medians, means and scales are exported with numeric weights.
NPZ inference avoids unpickling Python objects. SHA-256 binds weights to the JSON report;
this protects against accidental corruption, not a malicious actor replacing both files.

## Initial training data and performance

The bootstrap catalog requests 10 public Wikimedia scalp photographs, 5 each under
alopecia areata and tinea capitis source labels. The actual counts and failures are
recorded in the checked-in `ingestion_report.json` and `report.json`. Labels were
selected from source descriptions and have not been clinically adjudicated. Patient
identity is unknown; photos may depict repeated patients. All samples are train-only.
The paired clinical dataset is empty. Initial visual review excluded a miscategorized
nonhuman photograph and an ambiguously described photo before the published training
run. The exclusion record is retained in the ingestion report.

**No diagnostic accuracy, sensitivity, specificity, calibration or generalization
claim is supported.** Training loss is an optimization trace, not validation. There
is no healthy control, other-condition class, external site or patient-held-out test.
Reviewed research mode emits basic held-out metrics but those do not constitute
clinical validation. Patient-level global-metric bootstrap intervals are available for reviewed cohorts;
see [evaluation methodology](EVALUATION.md) for assumptions and small-cohort guards.
Calibration, subgroup performance and prospective study results remain future work.

## Failure modes

- May learn skin tone, camera, source, background, annotations or compression artifacts.
- Cannot distinguish unseen conditions or safely recognize a healthy scalp.
- May produce confident scores for non-scalp photographs or unfamiliar modalities.
- Limited to source labels in its particular training artifact.
- Single-label outputs omit coexisting disorders and diagnostic ambiguity.
- Basic resolution/exposure/contrast gates do not verify focus or clinical suitability.
- Trichoscopy and ordinary scalp photography have different domains; evaluate separately.
- Missing or unreliable labs and findings can distort multimodal research results.

Outputs always abstain and mark `clinically_validated: false`; there is no switch to
silently enable clinical diagnosis. A future validated release needs a new reviewed
protocol and implementation, not an arbitrary confidence threshold.

## Privacy and deployment

The local analysis endpoint persists neither photos nor clinical requests and disables
access logs in the CLI server. No third-party model API receives patient data. Training
does read locally curated data, so it must run in an appropriately protected environment.
Public artifacts contain no paired clinical records. Never add private run manifests to
this repository. Authentication, role controls, consent workflows, encryption at rest,
retention policies and clinical audit trails are not implemented.

