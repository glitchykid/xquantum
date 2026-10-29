# Verification: public photo to research response

Date: 2026-10-30. Runtime: Python 3.12.14 on CPU. The user story is: a researcher
downloads explicitly cataloged public scalp photos, audits the dataset, trains a
model, then submits a photo and structured context through the CLI or local HTTP API.
There is no graphical interface or external model-service dependency.

| Boundary | Result | Evidence |
|---|---|---|
| Public sources → local data | Passed | 10 accepted photos after 2 source/visual exclusions; original source, author, license and checksums retained |
| Manifest → audit | Passed | Source labels kept unreviewed, null patient IDs, train-only; checksums and duplicates checked |
| Data → training | Passed | 50 real optimization epochs, seed 42, 5 samples per class; per-epoch loss in report.json |
| Training → saved artifact | Passed | Numeric NPZ weights and JSON report, SHA-256 verification, no executable object deserialization |
| Installed CLI → model | Passed | Actual downloaded image analyzed using saved weights and fictional clinical context |
| Local HTTP → model → response | Passed | HTTP 200 from /v1/analyze, status abstain, two research scores, Cache-Control no-store |
| Invalid request → error | Passed | TestClient covers corrupt base64/images, invalid clinical values and oversized body |
| Packaging → installable build | Passed | Editable install, wheel and source distribution built; pip check passed |

`pytest`: 23 passed, 96.85% statement coverage. `ruff check .` and
`ruff format --check src tests` passed. The first test invocation failed because
the application modules did not yet exist, before implementation began.

The final bootstrap has no held-out test results because source images lack verified
patient identities. Generated software test fixtures are not medical training data.
No prospective or external clinical evaluation, regulatory assessment, production
deployment, interface/browser evaluation or real paired-medical-data training was run.

Original photographs and the rejected exploratory run remain outside version control.
Public release artifacts contain no paired clinical records. The example response
uses fictional clinical context and contains no image bytes or patient identifiers.
