# XQuantum

Python research software for hair and scalp image analysis, intended for development
with trichologists and dermatologists. The first release is a working CPU baseline,
not a diagnostic medical device. **Every inference abstains from diagnosis.**

It downloads a small explicit catalog of openly licensed internet photos and trains
a real classifier. It also supports independently curated, specialist-reviewed
datasets with paired clinical features. No comprehensive clinical dataset is included
or claimed: building that dataset is the next research milestone.

## Implemented

- JPEG/PNG/WEBP decoding with byte, pixel, format and size limits; EXIF orientation handling.
- License allowlist, original source links, attribution, checksums, metadata stripping,
  and explicit `source_only` labels for Wikimedia bootstrap photos.
- Manifest auditing for checksums, exact duplicates, possible perceptual duplicates
  across splits, patient overlap, and source-group overlap.
- Reproducible class-balanced logistic training with SGD, training-only preprocessing,
  deterministic seed, per-epoch loss, numeric NPZ weights, and JSON experiment reports.
- Optional early fusion of image features and measured clinical values with missingness
  indicators. Unknown findings stay unknown. Training rejects invented/unpaired bootstrap data.
- Local FastAPI endpoint and CLI sharing the same inference implementation. Uploaded
  images and clinical inputs are processed in memory and are not retained.
- Reviewed research runs report held-out balanced accuracy, macro F1, log loss,
  class support, and confusion matrices. Bootstrap runs report **no test accuracy**.
- Primary patient-level evaluation averages views per patient, with per-condition
  sensitivity/specificity and seeded patient-bootstrap confidence intervals. See
  [evaluation methodology](docs/EVALUATION.md).

## Installation

Use Python 3.12 or newer and a CPU. No GPU, paid API, cloud account, or model key is needed.
Internet access is needed for package installation and the Wikimedia downloader.
Core package versions were checked against PyPI and official documentation on
2026-10-30. Exact installed dependencies are also recorded in `requirements-lock.txt`.

```bash
git clone https://github.com/glitchykid/XQuantum.git
cd XQuantum
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

For development, the lock includes test, formatting, and build tools. To resolve a
fresh environment from direct requirements instead, use `python -m pip install -e '.[dev]'`.

## Run the included baseline

The checked-in `artifacts/bootstrap/` contains the initial run's weights, report,
manifest snapshot and attribution. Original images stay out of Git.

```bash
xquantum fetch-bootstrap --output data/commons
xquantum audit --manifest data/commons/manifest.jsonl --bootstrap
xquantum train --manifest data/commons/manifest.jsonl --bootstrap \
  --output artifacts/my-bootstrap --epochs 50 --seed 42
xquantum analyze --model artifacts/my-bootstrap \
  --image /absolute/path/to/scalp.jpg --clinical examples/clinical.json
xquantum serve --model artifacts/my-bootstrap --port 8000
```

Use a new output directory for each run. The downloader preserves source attribution
and refuses to overwrite an existing manifest. Inspect `ingestion_report.json` for
failed downloads. Wikimedia files may change or disappear; checksums describe the
exact downloaded snapshot, not a promise that a future download is byte-identical.

## API

The server binds to **127.0.0.1**. `GET /health` reports model availability.
`GET /openapi.json` exposes the machine-readable API schema; no graphical UI is included.
`POST /v1/analyze` accepts a base64 image and optional structured clinical context:

```python
import base64
import json
import urllib.request
from pathlib import Path

payload = {
    "image_base64": base64.b64encode(Path("scalp.jpg").read_bytes()).decode(),
    "clinical": {"age_years": 35, "duration_months": 6, "scaling": True},
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/analyze",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))
```

Results contain `status: "abstain"`, quality issues, model identity, observed findings
for specialist review, and experimental `research_scores`. Scores sum to one but are
**uncalibrated model outputs, not probabilities that a patient has a disease**.
Poor resolution, extreme exposure or very low contrast suppress the scores. These
are simple checks, not validated image-quality, scalp or out-of-distribution detectors.
Image-only models explicitly report `clinical_context_used_by_model: false`.

## Multi-photo case analysis

Use `xquantum analyze-case --model artifacts/bootstrap --image vertex.jpg temporal.jpg`
or `POST /v1/analyze-case` with `images_base64` and optional `clinical` context.
The case averages all views and reports per-view quality; any quality failure
suppresses combined scores. See [case analysis](docs/CASE_ANALYSIS.md) for limits.

## Restricted and non-free sources

Authorized non-free datasets are supported with an explicit use scope, permission
reference and redistribution flag. Training reports flag restricted-source models
for release review. See [restricted data](docs/RESTRICTED_DATA.md) for fields and
assessed providers. No restricted images have been added to the public bootstrap.

## Train with reviewed medical data

See [data contract](docs/DATA_CONTRACT.md) for the JSONL format, units, labels,
annotation rules and protected-data handling. JSON schemas can be exported with:

```bash
xquantum schema --kind clinical > clinical.schema.json
xquantum schema --kind sample > sample.schema.json
xquantum audit --manifest /secure/dataset/manifest.jsonl
xquantum train --manifest /secure/dataset/manifest.jsonl --multimodal \
  --output /secure/runs/reviewed-001 --epochs 50 --seed 42
```

Reviewed runs require explicit train/validation/test splits, known patient groups,
known image modality, source rights, reviewer identifiers, and at least two training
classes. Multimodal runs additionally require at least two nonempty paired clinical
records per class. That is an engineering minimum, not an adequate sample size for
clinical validation. Omit `--multimodal` for reviewed image-only experiments.
The fixed epoch count does not tune on the test set. Do not repeatedly select models
using test results. The audit cannot discover every re-upload or unknown patient.

## Layout

```text
src/xquantum/       schemas, image handling, ingestion, auditing, model, CLI, API
tests/             unit, data-integrity, training round-trip and API tests
examples/          fictional clinical example (never a real patient's record)
docs/              data contract, model card, research protocol, sources, release notes
docs/wiki/         version-controlled wiki documentation
data/              local downloaded or curated datasets (ignored by Git)
artifacts/         experiment outputs; only the public bootstrap snapshot is tracked
.github/workflows/ reproducible CI checks
```

## Validate and build

```bash
ruff check .
ruff format --check src tests
pytest
python -m build
python -m pip check
```

`pytest` runs offline tests and enforces at least 85% statement coverage. Tests use
generated fixtures, which validate software behavior and are not medical training data.
CI does not download patient photos. To format code, use `ruff format src tests`.

## Scope and next milestone

The public bootstrap contains ten photos and distinguishes only source-labeled alopecia areata and tinea
capitis images. It has no verified healthy-control class, no known patient count,
no clinical records, and no defensible held-out evaluation. It cannot rule out any
condition, identify an underlying cause, determine disease severity or prescribe treatment.
The wider ten-label taxonomy is a data contract, not evidence of trained coverage.

The next milestone is a consented, diverse, specialist-adjudicated cohort with
standardized scalp and trichoscopy views and paired history. Follow
[research protocol](docs/RESEARCH_PROTOCOL.md) before comparing stronger image
encoders or making clinical performance claims. No CNN or pretrained medical
foundation model is included in this release; the baseline deliberately establishes
the complete working data-to-inference path first.

The local API has no authentication, encrypted record storage, or clinical audit
system. Keep it local and use authorized, de-identified research data. Hosting a
clinical service needs those controls, external validation and appropriate local
regulatory review. Do not commit private manifests, patient IDs, lab results or photos.

Clinical motivation and source limitations are documented in [sources](docs/SOURCES.md).
`AGENTS.md` is an unchanged copy of the user-selected upstream repository guidelines.

