# Development update — multi-photo cases and restricted data

Date: 2026-10-30. Source update; release automation permissions are unchanged.

Added `analyze-case` and `/v1/analyze-case`, shared per-view preprocessing, bounded
case inputs, duplicate detection and all-view quality handling. Added restricted-use
manifest fields and training report flags. Non-free provider terms were assessed;
no paid dataset was purchased or non-free image cohort acquired. Existing public
weights and diagnostic abstention behavior remain unchanged.

# Development update — patient-level evaluation

Date: 2026-10-30. Published as a source update; no new release tag or clinical claim.

Reviewed runs now include per-condition sensitivity/specificity, equal-patient
view aggregation and deterministic 95% bootstrap intervals for global metrics.
Missing classes and insufficient patients produce explicit warnings. Bootstrap
weights and abstention behavior are unchanged. See `EVALUATION.md` for assumptions.

# v0.1.0 — Runnable research baseline

Date: 2026-10-30

Initial Python package for specialist-led hair/scalp research. Includes a working
local API and CLI, a provenance-aware public-image downloader, manifest integrity
and leakage auditing, reproducible CPU classifier training, optional paired clinical
fusion, numeric model export, documentation and CI/release workflows.

## Training actually completed

- 50 epochs with seed 42 on 10 real, openly licensed internet photos.
- Source labels: 5 alopecia areata and 5 tinea capitis images.
- A nonhuman file and a source-ambiguous file were excluded after visual/source review.
- The published weights, training history, manifest and attribution are under
  `artifacts/bootstrap/`. Original photos and excluded-run weights are not committed.
- There are no paired medical records and no held-out clinical metrics.

## Validation

- 23 offline tests passed, 96.85% statement coverage (85% required).
- Ruff lint and formatting checks passed.
- Wheel and source distribution built successfully; dependency consistency passed.
- Installed CLI and live localhost HTTP request tested with an actual downloaded photo.
- Model reload, corruption rejection, patient/source/duplicate leakage checks,
  multimodal test fixtures, invalid input and oversized requests covered.

See `docs/VERIFICATION.md` for the concrete flow and artifact details. Local software
checks establish implementation behavior, not diagnostic performance. GitHub Actions
repeats the checks; its actual run status is reported by GitHub.

## Limitations

This is a **research-only baseline**, not a clinically validated diagnostic system.
The model always abstains from diagnosis. Scores are uncalibrated; unknown patient
identities prevent a defensible bootstrap evaluation. There is no comprehensive
medical cohort, healthy-control class, CNN, validated scalp detector, clinical
authentication or patient-record system. Wider clinical data collection and external
validation remain required. See the model card and research protocol.

Raw internet images retain their per-file licenses. Source metadata and attribution
are provided for reproducibility; check rights before any further reuse.
