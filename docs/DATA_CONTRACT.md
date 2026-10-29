# Data contract

One JSON object per line in `manifest.jsonl`. Paths resolve relative to that file
and may not escape its directory, including through symlinks. All images must be
JPEG, PNG or WEBP, 64 pixels or larger in each dimension, at most 10 MiB and 20 MP.
Use standardized, unannotated scalp photographs and trichoscopy; do not mix diagrams,
publication panels, treatment comparisons, hair-product ads or identity portraits.

## Fields

| Field | Contract |
|---|---|
| `image_path` | Relative image path within the manifest directory |
| `label` | One of the labels below; a single adjudicated primary label |
| `patient_id` | Pseudonymous patient group; required for reviewed research |
| `source_group` | Related encounter/case/publication/re-upload group; keep in one split |
| `source_url` | HTTP(S) provenance location; do not include signed URLs or identifiers |
| `license` | License or authorized institutional-use agreement |
| `rights_basis` | Evidence supporting this research use; no PHI or secrets |
| `split` | Explicit `train`, `validation` or `test` |
| `label_status` | `source_only` or `specialist_reviewed` |
| `reviewer_id` | Pseudonymous accountable specialist; required for reviewed labels |
| `photo_type` | `scalp`, `trichoscopy`, or `unknown`; known for reviewed research |
| `sha256` | SHA-256 of the exact file bytes, lowercase hex |
| `clinical` | Optional structured record; missing values are `null`, not zero |

Bootstrap records have null patient IDs and stay entirely in `train`. The entire
Wikimedia cohort has a shared conservative `source_group`; no patient-independent
split is inferred from filenames. Internet labels are never promoted automatically.
The software checks field completeness, not the legal sufficiency of an agreement
or the truth of a specialist attestation.

## Taxonomy

`alopecia_areata`, `androgenetic_alopecia`, `telogen_effluvium`, `traction_alopecia`,
`scarring_alopecia`, `tinea_capitis`, `scalp_psoriasis`, `seborrheic_dermatitis`,
`folliculitis`, `unaffected`.

The first trained artifact includes only two labels. Coexisting disease is common;
this initial classifier's single-label contract cannot represent it. Exclude ambiguous
cases from baseline experiments, record adjudication externally, and design a
multi-label protocol before broadening the clinical task.

## Clinical fields

Numerical fields: `age_years` (0–120), `duration_months` (0–1440),
`skin_phototype` (integer 1–6). Optional `sex_at_birth` is `female`, `male`,
`intersex`, or `unknown`. It should be collected only when justified by the study.
Skin phototype is not an ethnicity label or a complete measure of skin tone.

Optional findings: `itching`, `pain`, `scaling`, `pustules`, `sudden_shedding`,
`family_history`, `recent_illness`, `postpartum`, `traction_exposure`,
`loss_of_follicular_openings`, `miniaturization`, `yellow_dots`, `black_dots`,
`broken_hairs`, `perifollicular_erythema`. Values are true, false or null.

Optional `labs` uses explicit names and units:

| Field | Unit |
|---|---|
| `ferritin_ng_ml` | ng/mL |
| `hemoglobin_g_dl` | g/dL |
| `tsh_miu_l` | mIU/L |
| `vitamin_d_ng_ml` | ng/mL |
| `zinc_ug_dl` | µg/dL |

Numeric bounds reject impossible or extreme entry mistakes; they are **not clinical
reference intervals**. Do not order these tests routinely because the schema lists
them. Their relevance, reference ranges and interpretation depend on the clinician
and context. Store collection dates, laboratory reference ranges, longitudinal
history, medications and biopsy/mycology evidence in the authorized source system;
this prototype does not replace a comprehensive medical record.

The model consumes the numerical fields, Boolean findings, laboratory values, three
sex indicators and missingness indicators only when trained with `--multimodal`.
All imputation and scaling are fitted on training data alone. Split assignments,
patient/reviewer/source IDs, licenses and labels never become model features.
Image-only bootstrap weights do not infer lab values or learn from clinical context.

## Annotation and splits

Obtain documented research permission and applicable subject consent. Strip EXIF/GPS
before ingestion and inspect for visible identifiers. Use independent specialist
review, retain diagnostic evidence in the protected source system, and adjudicate
disagreements. Exclude identifiers, watermarks, captions and post-treatment comparisons
that leak the label. Freeze the cohort and test set before model development.
Group all visits/views from a patient, linked cases and duplicates together.

The exact and perceptual hash checks are conservative screening tools. They do not
replace manual duplicate review or source reconciliation. A `source_group` should
represent related samples, not automatically an entire hospital; leave whole sites
out separately for external validation. No image-level random-split helper is provided.

