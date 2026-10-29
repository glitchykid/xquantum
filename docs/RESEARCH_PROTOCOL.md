# Research protocol and development gates

## Working slice delivered

The first milestone establishes an actual internet-photo-to-trained-artifact-to-API
path. Its small transparent CPU classifier is a comparator for future models, not a
substitute for a clinical cohort. Software fixtures are isolated from medical data.

## Data collection with specialists

Define an intended population, acquisition protocol, diagnostic scope and exclusion
criteria before collection. Capture standardized vertex/frontal/temporal/occipital
views and trichoscopy where justified, linked to a pseudonymous patient. Record
onset, pattern, symptoms, history, exposures, medications and indicated investigations
in a protected clinical system. Include unaffected controls and competing diagnoses.
Do not require investigations solely to populate machine-learning fields.

Agree on reference diagnoses using independent specialist review and adjudication;
include pathology or mycology when clinically appropriate. Preserve uncertainty and
coexisting disease. Obtain appropriate ethics, institutional permission, consent,
retention and dataset-use terms. A public-image license alone does not establish
fitness for clinical validation or resolve every subject/privacy right.

## Cohort and analysis design

Predefine sample-size and precision targets with a statistician. Recruit across
clinics, devices, age groups, hair types, skin tones and disease severity. Avoid
simulating missing patient characteristics or inferring identity/demographics from
images. Freeze patient-level splits; reserve independent external sites and time periods.
Have specialists review deduplication and leakage reports before model fitting.

Use the shipped image-only classifier as the baseline. Next compare pretrained
image encoders and carefully tested clinical fusion on the same fixed cohort.
Separate photo modalities or model them explicitly. Evaluate the incremental value
of clinical fields and perform missing-data and source-confounding ablations.
Choose hyperparameters on development data and calibration on a separate calibration
set; do not use the final test cohort for either task.

## Evaluation before clinical consideration

Report patient counts, per-condition sensitivity/specificity, precision/recall, macro
F1, discrimination and calibration, with patient-clustered confidence intervals.
Assess age/skin-tone/hair-type/device/site subgroups, unknown conditions, poor images,
out-of-distribution inputs and abstention coverage/error tradeoffs. Avoid multiple
photos from one patient inflating effective test size. Basic metrics emitted by this
release are an engineering starting point, not this full statistical analysis.

Perform external validation and a prospective study of specialist-assisted use,
including failure review and measurable clinical benefit. Plan clinical responsibility,
monitoring, incident handling, privacy controls and the applicable regulatory pathway.
Do not present source-photo training performance as patient benefit.

